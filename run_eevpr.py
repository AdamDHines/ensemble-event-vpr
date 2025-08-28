import tensorflow as tf

import itertools
import scipy
import scipy.spatial

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import os
from os import listdir, symlink
from pathlib import Path

import numpy as np

from tqdm.auto import tqdm

from code_helpers_public import get_timestamps, get_timestamp_matches, get_image_sets_on_demand, get_vlad_features
from read_gps import get_gps
from correspondence_event_camera_frame_camera import traverse_to_name, name_to_consumervideo, video_beginning

import argparse, json, sys

tqdm.get_lock().locks = []

def get_image_paths(folder1):
    return sorted([os.path.join(folder1, f) for f in listdir(folder1) if f.endswith('.png')])

def load_e2vid_abs_times(metadata_json_path, timestamps_txt_path):
    """Return absolute frame times (seconds) from metadata.json + e2vid timestamps.txt."""
    with open(metadata_json_path, 'r') as f:
        md = json.load(f)
    if 'start_time_ns' not in md:
        raise KeyError(f"{metadata_json_path} missing 'start_time_ns'")
    start_s = float(md['start_time_ns']) * 1e-9
    # timestamps.txt contains per-frame times in seconds (E2VID output)
    ts = np.loadtxt(timestamps_txt_path, dtype=float)
    abs_ts = start_s + ts
    if not np.all(np.diff(abs_ts) >= 0):
        abs_ts = np.sort(abs_ts)
    return abs_ts

# --- helpers ---
def _abs_times_for_dir(frames_dir: str) -> np.ndarray:
    ts_path   = os.path.join(frames_dir, 'timestamps.txt')
    meta_path = os.path.join(Path(frames_dir).parent, 'metadata.json')
    ts = load_e2vid_abs_times(str(meta_path), str(ts_path))  # seconds, sorted
    return ts.astype(float)

def _pair_canonical(q_times: np.ndarray, r_times: np.ndarray, max_dt: float):
    """
    Robust time pairing between two streams:
      1) fit linear map r ≈ a*q + b,
      2) nearest-neighbour under a cadence-based tolerance,
      3) enforce monotonicity,
      4) one fallback with a looser tol.
    Returns (t_q, t_r) with equal length.
    """
    q = np.asarray(q_times, float)
    r = np.asarray(r_times, float)
    if q.size == 0 or r.size == 0:
        return q[:0], r[:0]
    q.sort(); r.sort()

    # --- 1) linear time map r ≈ a*q + b (handles different window cadences) ---
    if q[-1] > q[0]:
        a = (r[-1] - r[0]) / (q[-1] - q[0])
    else:
        a = 1.0
    b = r[0] - a * q[0]
    r_pred = a * q + b

    # cadence-based tolerance (use median step); don’t trust tiny fixed max_dt
    dq = (np.median(np.diff(q)) if q.size > 1 else max_dt)
    dr = (np.median(np.diff(r)) if r.size > 1 else max_dt)
    tol = max(max_dt, 0.6 * max(dq, dr))

    def _nn_monotonic(q_hat, r_arr, tol_):
        idx = np.searchsorted(r_arr, q_hat, side='left')
        best = np.full(q_hat.size, -1, dtype=int)
        for i, (t, j) in enumerate(zip(q_hat, idx)):
            c = []
            if j > 0:           c.append(j - 1)
            if j < r_arr.size:  c.append(j)
            if not c:           continue
            c = np.asarray(c, int)
            k = int(c[np.argmin(np.abs(r_arr[c] - t))])
            if abs(r_arr[k] - t) <= tol_:
                best[i] = k
        # monotonic non-decreasing
        out = np.full_like(best, -1)
        prev = -1
        for i, j in enumerate(best):
            if j < 0: continue
            if j >= prev:
                out[i] = j; prev = j
            elif prev >= 0 and abs(r_arr[prev] - q_hat[i]) <= tol_:
                out[i] = prev
        valid = out >= 0
        return valid, out

    # --- 2) NN + monotonic with cadence tol ---
    valid, out = _nn_monotonic(r_pred, r, tol)

    # --- 3) fallback once with looser tol if empty ---
    if not np.any(valid):
        tol2 = max(tol * 2.0, 1.5 * max(dq, dr), 0.25)
        valid, out = _nn_monotonic(r_pred, r, tol2)
        if not np.any(valid):
            return q[:0], r[:0]

    return q[valid], r[out[valid]]

def main():
    parser = argparse.ArgumentParser()

    # Input arguments
    parser.add_argument('--dataset', '-ds', type=str, help='Dataset name',
                        default="qcr_event")
    parser.add_argument('--reference', '-r', type=str, help='Reference traverse name',
                        default="traverse1")
    parser.add_argument('--query', '-q', type=str, help='Query traverse name',
                        default="traverse2")
    parser.add_argument('--window_duration', '-w', type=list, help='Fixed window duration in ms',
                        default=[500, 1000])
    parser.add_argument('--num_events_per_pixel', '-n', type=list, help='Number of events per pixel',
                        default=[1.0])
    parser.add_argument('--gps_available', action='store_true', help='Whether GPS data is available',
                        default=False)
    parser.add_argument('--gps_format', type=str, help='GPS format (nmea or kml)',
                        default='nmea')
    
    # Directories
    parser.add_argument('--dataset_folder', '-d', type=str, help='Path to dataset folder',
                        default="/Users/adam/repo/Event-LAB/datasets/qcr_event")
    parser.add_argument('--netvlad_folder', '-nv', type=str, help='Path to NetVLAD folder',
                        default="/Users/adam/repo/netvlad_tf_open")
    parser.add_argument('--frames_subfolder', type=str, help='Subfolder name for frames',
                        default='reconstruction')
    parser.add_argument('--eventlab_dir', type=str, help='Path to Event-LAB folder',
                        default="/Users/adam/repo/Event-LAB")
    
    # Parse the arguments
    args = parser.parse_args()

    # Append NetVLAD folder to system path
    sys.path.append(os.path.join(args.netvlad_folder, 'python'))
    import netvlad_tf.nets as nets

    # Get the reference and query directories for time windows and number of events per pixel
    reference_window_dirs = [os.path.join(f'{args.dataset_folder}',f'{args.reference}',f'{args.reference}-{args.frames_subfolder}-{timewindow}',f'{args.frames_subfolder}') for timewindow in args.window_duration]
    reference_count_dirs = [os.path.join(f'{args.dataset_folder}',f'{args.reference}',f'{args.reference}-{args.frames_subfolder}-{num_events_per_pixel}',f'{args.frames_subfolder}') for num_events_per_pixel in args.num_events_per_pixel]
    reference_combined_dirs = reference_window_dirs + reference_count_dirs

    query_window_dirs = [os.path.join(f'{args.dataset_folder}',f'{args.query}',f'{args.query}-{args.frames_subfolder}-{timewindow}',f'{args.frames_subfolder}') for timewindow in args.window_duration]
    query_count_dirs = [os.path.join(f'{args.dataset_folder}',f'{args.query}',f'{args.query}-{args.frames_subfolder}-{num_events_per_pixel}',f'{args.frames_subfolder}') for num_events_per_pixel in args.num_events_per_pixel]
    query_combined_dirs = query_window_dirs + query_count_dirs

    # Get the image paths for both reference and query directories
    reference_paths = {}
    query_paths = {}

    for subfolder in reference_combined_dirs:
        reference_paths[subfolder] = get_image_paths(os.path.join(args.dataset_folder, subfolder))

    for subfolder in query_combined_dirs:
        query_paths[subfolder] = get_image_paths(os.path.join(args.dataset_folder, subfolder))
    
    # Calculate the equivelancies across timewindows
    if args.gps_available:
        timestamps_combined1 = {}; timestamps_combined2 = {}

        for subfolder in all_subfolders:
            if subfolder == 'frames':
                timestamps_combined1[subfolder] = np.array([float(os.path.splitext(basename(f))[0]) for f in image_paths_combined1[subfolder]])
                timestamps_combined2[subfolder] = np.array([float(os.path.splitext(basename(f))[0]) for f in image_paths_combined2[subfolder]])
            else:
                timestamps_combined1[subfolder], timestamps_combined2[subfolder] = get_timestamps(join(dataset_folder, subfolder, query_traverse), join(dataset_folder, subfolder, reference_traverse))
        x1 = get_gps(vid_path_1 + '.nmea')
        x2 = get_gps(vid_path_2 + '.nmea')

        match_x1_to_x2 = []
        for idx1, (latlon, t) in enumerate(zip(x1[:, 0:2], x1[:, 2])):
            if len(match_x1_to_x2) < 6:
                min_idx2 = 0
                max_idx2 = int(0.25 * len(x2))
            elif idx1 > 0.5 * len(x1):
                min_idx2 = match_x1_to_x2[-5]
                max_idx2 = len(x2)
            else:
                min_idx2 = match_x1_to_x2[-5]
                max_idx2 = int(0.75 * len(x2))
            best_match = (np.linalg.norm(x2[min_idx2:max_idx2, 0:2] - latlon, axis=1)).argmin() + min_idx2
            match_x1_to_x2.append(best_match)
        match_x1_to_x2 = np.array(match_x1_to_x2)

        t_raw1 = x1[:, 2]
        t_raw2 = x2[match_x1_to_x2, 2]
        timestamps_gps1 = np.array([t + video_beginning[traverse_to_name[query_traverse]] for t in t_raw1])
        timestamps_gps2 = np.array([t + video_beginning[traverse_to_name[reference_traverse]] for t in t_raw2])

        images_all_combined_set1 = {}; images_all_combined_set2 = {}
        matches_fixedlength_combined1 = {}; matches_fixedlength_combined2 = {}

        for subfolder in all_subfolders:
            matches_fixedlength_combined1[subfolder] = get_timestamp_matches(timestamps_combined1[subfolder], timestamps_gps1)
            matches_fixedlength_combined2[subfolder] = get_timestamp_matches(timestamps_combined2[subfolder], timestamps_gps2)
            images_all_combined_set1[subfolder], images_all_combined_set2[subfolder] = \
                get_image_sets_on_demand(image_paths_combined1[subfolder], image_paths_combined2[subfolder], 
                                        matches_fixedlength_combined1[subfolder], matches_fixedlength_combined2[subfolder])
    else:
        images_all_combined_set1 = {}
        images_all_combined_set2 = {}
        matches_fixedlength_combined1 = {}
        matches_fixedlength_combined2 = {}

        max_dt = getattr(args, 'max_dt', 0.06)

        # ---- Time-window pairs ----
        for q_dir, r_dir in zip(query_window_dirs, reference_window_dirs):
            q_times = _abs_times_for_dir(q_dir)
            r_times = _abs_times_for_dir(r_dir)
            t_q, t_r = _pair_canonical(q_times, r_times, max_dt)  # canonical paired timelines

            # attach each stream (dir) to its canonical timeline
            matches_fixedlength_combined1[q_dir] = get_timestamp_matches(q_times, t_q)
            matches_fixedlength_combined2[r_dir] = get_timestamp_matches(r_times, t_r)

            iq, ir = get_image_sets_on_demand(
                query_paths[q_dir],         # list of query image paths (q_dir)
                reference_paths[r_dir],     # list of reference image paths (r_dir)
                matches_fixedlength_combined1[q_dir],
                matches_fixedlength_combined2[r_dir]
            )
            images_all_combined_set1[q_dir] = iq    # key is the query subfolder path
            images_all_combined_set2[r_dir] = ir    # key is the reference subfolder path

        # ---- Count (events-per-pixel) pairs ----
        for q_dir, r_dir in zip(query_count_dirs, reference_count_dirs):
            q_times = _abs_times_for_dir(q_dir)
            r_times = _abs_times_for_dir(r_dir)
            t_q, t_r = _pair_canonical(q_times, r_times, max_dt)

            matches_fixedlength_combined1[q_dir] = get_timestamp_matches(q_times, t_q)
            matches_fixedlength_combined2[r_dir] = get_timestamp_matches(r_times, t_r)

            iq, ir = get_image_sets_on_demand(
                query_paths[q_dir],
                reference_paths[r_dir],
                matches_fixedlength_combined1[q_dir],
                matches_fixedlength_combined2[r_dir]
            )
            images_all_combined_set1[q_dir] = iq
            images_all_combined_set2[r_dir] = ir

    # === Feature extraction using NetVLAD (keys are actual dir paths) ===
    comparison_id = f"{args.query}_{args.reference}"
    overall_data_dir = os.path.join(args.dataset_folder, "overall_data", comparison_id)
    os.makedirs(overall_data_dir, exist_ok=True)

    # Init TF graph/session
    tf.compat.v1.disable_eager_execution(); tf.compat.v1.reset_default_graph()
    image_batch = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, None, None, 3])
    net_out = nets.vgg16NetvladPca(image_batch)
    saver = tf.compat.v1.train.Saver()

    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.5, allow_growth=True)
    config = tf.compat.v1.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1, gpu_options=gpu_options)
    config.log_device_placement = True
    sess = tf.compat.v1.Session(config=config)
    saver.restore(sess, nets.defaultCheckpoint())
    print("NetVLAD ready")

    def _safe_pair_dir(q_dir: str, r_dir: str) -> str:
        q_rel = os.path.relpath(q_dir, args.dataset_folder).replace(os.sep, "__")
        r_rel = os.path.relpath(r_dir, args.dataset_folder).replace(os.sep, "__")
        return os.path.join(overall_data_dir, "pairs", f"{q_rel}__VS__{r_rel}")

    features_all_combined1 = {}   # keyed by query dir path
    features_all_combined2 = {}   # keyed by reference dir path
    dist_matrix_all_combined = {} # keyed by (q_dir, r_dir) tuple

    total_pairs = len(query_window_dirs) + len(query_count_dirs)
    with tqdm(total=2*total_pairs, position=0, leave=False) as pbar:

        # ---- time-window pairs ----
        for q_dir, r_dir in zip(query_window_dirs, reference_window_dirs):
            imgs_q = images_all_combined_set1[q_dir]
            imgs_r = images_all_combined_set2[r_dir]

            pair_dir = _safe_pair_dir(q_dir, r_dir)
            os.makedirs(pair_dir, exist_ok=True)
            f1_path = os.path.join(pair_dir, "netvlad_features_all_set1.npy")
            f2_path = os.path.join(pair_dir, "netvlad_features_all_set2.npy")

            # handle empties
            if len(imgs_q) == 0 or len(imgs_r) == 0:
                features_all_combined1[q_dir] = np.zeros((0, 4096), dtype=np.float32)
                features_all_combined2[r_dir] = np.zeros((0, 4096), dtype=np.float32)
                dist_matrix_all_combined[(q_dir, r_dir)] = np.zeros((len(imgs_q), len(imgs_r)), dtype=np.float32)
                continue

            features_all_combined1[q_dir] = get_vlad_features(sess, net_out, image_batch, imgs_q, f1_path); pbar.update(1)
            features_all_combined2[r_dir] = get_vlad_features(sess, net_out, image_batch, imgs_r, f2_path); pbar.update(1)

            # Distance matrix (Q x R)
            dist_matrix_all_combined[(q_dir, r_dir)] = scipy.spatial.distance.cdist(
                features_all_combined1[q_dir], features_all_combined2[r_dir], metric="cosine"
            )

            # sanity
            assert dist_matrix_all_combined[(q_dir, r_dir)].shape == (len(imgs_q), len(imgs_r)), \
                f"Distance shape mismatch for pair {q_dir} vs {r_dir}"

        # ---- count (events-per-pixel) pairs ----
        for q_dir, r_dir in zip(query_count_dirs, reference_count_dirs):
            imgs_q = images_all_combined_set1[q_dir]
            imgs_r = images_all_combined_set2[r_dir]

            pair_dir = _safe_pair_dir(q_dir, r_dir)
            os.makedirs(pair_dir, exist_ok=True)
            f1_path = os.path.join(pair_dir, "netvlad_features_all_set1.npy")
            f2_path = os.path.join(pair_dir, "netvlad_features_all_set2.npy")

            if len(imgs_q) == 0 or len(imgs_r) == 0:
                features_all_combined1[q_dir] = np.zeros((0, 4096), dtype=np.float32)
                features_all_combined2[r_dir] = np.zeros((0, 4096), dtype=np.float32)
                dist_matrix_all_combined[(q_dir, r_dir)] = np.zeros((len(imgs_q), len(imgs_r)), dtype=np.float32)
                continue

            features_all_combined1[q_dir] = get_vlad_features(sess, net_out, image_batch, imgs_q, f1_path); pbar.update(1)
            features_all_combined2[r_dir] = get_vlad_features(sess, net_out, image_batch, imgs_r, f2_path); pbar.update(1)

            dist_matrix_all_combined[(q_dir, r_dir)] = scipy.spatial.distance.cdist(
                features_all_combined1[q_dir], features_all_combined2[r_dir], metric="cosine"
            )

            assert dist_matrix_all_combined[(q_dir, r_dir)].shape == (len(imgs_q), len(imgs_r)), \
                f"Distance shape mismatch for pair {q_dir} vs {r_dir}"


    # ### Ensemble stuff
    max_dist_plot = 20
    plot_threshold_fps = 6



    ensemble_init_kind = ensemble_init_kind_widget.value

    matchIndsCombined = []

    if ensemble_init_kind == 'zeros':
        print('Init with zeros')
        ensemble_init = np.zeros(dist_matrix_shape)
        init_len = 0
    elif ensemble_init_kind == 'frames':
        print('Include frames in ensemble')
        ensemble_init = dist_matrix_all_combined[frames_subfolder]
        init_len = 1
        matchIndsCombined.append(np.argmin(dist_matrix_all_combined[frames_subfolder], axis=0))
    else:
        raise ValueError('Wrong init type')

    dist_matrix_ensemble = ensemble_init.copy()

    for event_subfolder in event_subfolders:
        dist_matrix_ensemble += dist_matrix_all_combined[event_subfolder]

    for fixed_event_subfolder in fixed_event_subfolders:
        dist_matrix_ensemble += dist_matrix_all_combined[fixed_event_subfolder]

    dist_matrix_ensemble = dist_matrix_ensemble / (len(event_subfolders) + len(fixed_event_subfolders) + init_len)


    # In[29]:


    with tqdm(total=1 + len(all_subfolders), position=0, leave=False) as pbar:
        recalls_combined = {}; tps_combined = {}; best_matches_combined = {}
        recalls_best_individual = None; tps_best_individual=None; best_matches_individual=None

        recall_ensemble, tps_ensemble, best_matches_ensemble = get_recall_helper(dist_matrix_ensemble)
        pbar.update(1)

        for subfolder in all_subfolders:
            recalls_combined[subfolder], tps_combined[subfolder], best_matches_combined[subfolder] = get_recall_helper(dist_matrix_all_combined[subfolder])
            pbar.update(1)

            if subfolder!='frames' and (recalls_best_individual is None or recalls_combined[subfolder][plot_threshold_fps] > recalls_best_individual[plot_threshold_fps]):
                recalls_best_individual, tps_best_individual, best_matches_individual = recalls_combined[subfolder], tps_combined[subfolder], best_matches_combined[subfolder]
                best_individual_name = subfolder

    print('\n')
    print('Frame:                        %.3f' % recalls_combined['frames'][plot_threshold_fps])
    print('Best individual method        %.3f (%s)' % (recalls_best_individual[plot_threshold_fps], best_individual_name))
    print('Ensemble:                     %.3f' % recall_ensemble[plot_threshold_fps])


    # ### Plotting

    # In[30]:


    plt.plot(recall_ensemble, '-', label='Ensemble method', color='black')
    plt.plot(recalls_combined[frames_subfolder], '-', label='Frame-based')

    style=itertools.cycle(["-", "--", "-.", ":"])

    for event_subfolder in event_subfolders:
        plt.plot(recalls_combined[event_subfolder], ":")  # , label='Event-based ' + event_subfolder

    for fixed_event_subfolder in fixed_event_subfolders:
        plt.plot(recalls_combined[fixed_event_subfolder], ":")  # , label='Event-based ' + fixed_event_subfolder

    custom_line = Line2D([], [], color='brown', ls=':', label='Individual models')
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.append(custom_line)
    labels.append('Individual models')

    plt.xlabel("Time threshold [s]")
    plt.ylabel("Precision\n@ 100\% recall")
    plt.xlim(0, max_dist_plot)
    plt.ylim(0, 1.0)

    plt.savefig(join(path_to_plots, 'recall_curve_' + ensemble_init_kind + '.pdf'), bbox_inches='tight')
    plt.savefig(join(path_to_plots, 'recall_curve_' + ensemble_init_kind + '.pgf'), bbox_inches='tight')

    plt.show()
    plt.close()


    # In[34]:


    pr_ensemble = getPRCurveWrapper(dist_matrix_ensemble, plot_threshold_fps)
    pr_frames = getPRCurveWrapper(dist_matrix_all_combined['frames'], plot_threshold_fps)

    pr_event_combined = {}
    for subfolder in all_subfolders:
        pr_event_combined[subfolder] = getPRCurveWrapper(dist_matrix_all_combined[subfolder], plot_threshold_fps)


    # In[35]:


    print('PAt100R ensemble: %.3f' % getPAt100R(np.abs(np.argmin(dist_matrix_ensemble, axis=0)-np.arange(len(dist_matrix_ensemble))), plot_threshold_fps)[-1])


    # In[36]:


    print('getRAt99P ensemble: %.3f' % getRAt99P(pr_ensemble))


    # In[37]:


    print('PAt100R best    : %.3f' % getPAt100R(np.abs(np.argmin(dist_matrix_all_combined[best_individual_name], axis=0)-np.arange(len(dist_matrix_ensemble))), plot_threshold_fps)[-1])


    # In[38]:


    plt.plot(pr_ensemble[:, 1], pr_ensemble[:, 0], '-', label='Ensemble method', color='black')
    plt.plot(pr_event_combined[frames_subfolder][:, 1], pr_event_combined[frames_subfolder][:, 0], '-', label='Frame-based')

    style=itertools.cycle(["-", "--", "-.", ":"])

    for event_subfolder in event_subfolders:
        plt.plot(pr_event_combined[event_subfolder][:, 1], pr_event_combined[event_subfolder][:, 0], ":", label='Event-based ' + event_subfolder)

    for fixed_event_subfolder in fixed_event_subfolders:
        plt.plot(pr_event_combined[fixed_event_subfolder][:, 1], pr_event_combined[fixed_event_subfolder][:, 0], ":", label='Event-based ' + fixed_event_subfolder)

    custom_line = Line2D([], [], color='brown', ls=':', label='Individual models')
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.append(custom_line)
    labels.append('Individual models')

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim(0, 1.0)
    plt.ylim(0.0, 1.005)
    plt.legend()

    plt.savefig(join(path_to_plots, 'pr_curve_' + ensemble_init_kind + '.pdf'), bbox_inches='tight')

    plt.show()
    plt.close()


    # In[39]:


    avg_window_durations = []
    num_events_per_pixels = []
    xs_events, ys_events = [], []
    xs_events_window, ys_events_window = [], []
    xs_fixed_events, ys_fixed_events = [], []
    rat99ps = []

    perf_frames = recalls_combined[frames_subfolder][plot_threshold_fps]
    perf_ensemble = recall_ensemble[plot_threshold_fps]

    for event_subfolder in event_subfolders:
        perf = recalls_combined[event_subfolder][plot_threshold_fps]
        num_events_per_pixel = float(event_subfolder[event_subfolder.find('_')+1:])
        num_events_per_pixels.append(num_events_per_pixel)
        # TODO: Remove unique stuff which is/was for super low pixels per event
        avg_window_duration = (np.median(np.diff(np.unique(timestamps_combined1[event_subfolder]))) + np.median(np.diff(np.unique(timestamps_combined2[event_subfolder])))) / 2.0 * 1000.0
        avg_window_durations.append(avg_window_duration)
        xs_events.append(num_events_per_pixel); xs_events_window.append(avg_window_duration)
        ys_events.append(perf); ys_events_window.append(perf)
        rat99ps.append(getRAt99P(pr_event_combined[event_subfolder]))

    for fixed_event_subfolder in fixed_event_subfolders:
        perf = recalls_combined[fixed_event_subfolder][plot_threshold_fps]
        window_duration = float(fixed_event_subfolder[fixed_event_subfolder.find('_')+1:])
        xs_fixed_events.append(window_duration)
        ys_fixed_events.append(perf)
        rat99ps.append(getRAt99P(pr_event_combined[fixed_event_subfolder]))

    plt.ylabel("Recall (threshold=%d)" % (plot_threshold_fps-1))

    plt.ylim(0.0, 1.0)

    plt.axhline(perf_ensemble, linestyle='-', color='black', label='{:<32}- {:.3f}'.format('ensemble', perf_ensemble))

    plt.scatter([1000.0 / 30.0], [perf_frames], color='green', label='{:<32}- {:.3f}'.format('Frame-based approach', perf_frames))
    plt.plot(xs_events_window, ys_events_window, marker='D', markersize=7, label='{:<32}- {:.3f}'.format('Fixed num events; best', np.max(ys_events_window)))
    plt.plot(xs_fixed_events, ys_fixed_events, marker='*', markersize=10, label='{:<32}- {:.3f}'.format('Fixed time window; best', np.max(ys_fixed_events)))

    plt.xlabel("Window duration [milliseconds]")
    plt.legend(prop={'family': 'monospace'}, bbox_to_anchor=(0.96, -0.23))

    plt.savefig(join(path_to_plots, 'perf_joint.pdf'), bbox_inches='tight')

    plt.show()

if __name__ == "__main__":
    main()