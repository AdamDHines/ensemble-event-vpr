'''
Imports
'''
import os, argparse, sys, glob, scipy

import numpy as np
import tensorflow as tf

from tqdm import tqdm
from pathlib import Path
from read_gps import get_gps
from correspondence_event_camera_frame_camera import traverse_to_name, video_beginning
from code_helpers_public import get_timestamp_matches, get_image_sets_on_demand, get_vlad_features, get_image_paths, _abs_times_for_dir, _pair_canonical

def main():
    parser = argparse.ArgumentParser()

    # Input arguments
    parser.add_argument('--dataset', '-ds', type=str, help='Dataset name',
                        default="brisbane_event")
    parser.add_argument('--reference', '-r', type=str, help='Reference traverse name',
                        default="sunset2")
    parser.add_argument('--query', '-q', type=str, help='Query traverse name',
                        default="sunset1")
    parser.add_argument('--window_duration', '-w', type=list, help='Fixed window duration in ms',
                        default=[500])
    parser.add_argument('--num_events_per_pixel', '-n', type=list, help='Number of events per pixel',
                        default=[0.8])
    parser.add_argument('--gps_available', action='store_true', help='Whether GPS data is available',
                        default=False)
    parser.add_argument('--gps_format', type=str, help='GPS format (nmea or kml)',
                        default='nmea')
    
    # Directories
    parser.add_argument('--dataset_folder', '-d', type=str, help='Path to dataset folder',
                        default="/Users/adam/repo/Event-LAB/datasets")
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
    reference_window_dirs = [os.path.join(f'{args.dataset_folder}',f'{args.dataset}',f'{args.reference}',f'{args.reference}-{args.frames_subfolder}-{timewindow}',f'{args.frames_subfolder}') for timewindow in args.window_duration]
    reference_count_dirs = [os.path.join(f'{args.dataset_folder}',f'{args.dataset}',f'{args.reference}',f'{args.reference}-{args.frames_subfolder}-{num_events_per_pixel}',f'{args.frames_subfolder}') for num_events_per_pixel in args.num_events_per_pixel]
    reference_combined_dirs = reference_window_dirs + reference_count_dirs

    query_window_dirs = [os.path.join(f'{args.dataset_folder}',f'{args.dataset}',f'{args.query}',f'{args.query}-{args.frames_subfolder}-{timewindow}',f'{args.frames_subfolder}') for timewindow in args.window_duration]
    query_count_dirs = [os.path.join(f'{args.dataset_folder}',f'{args.dataset}',f'{args.query}',f'{args.query}-{args.frames_subfolder}-{num_events_per_pixel}',f'{args.frames_subfolder}') for num_events_per_pixel in args.num_events_per_pixel]
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
        # load GPS files from traverse roots
        reference_dir = os.path.join(args.dataset_folder, args.dataset, args.reference)
        query_dir     = os.path.join(args.dataset_folder, args.dataset, args.query)
        ref_gps_files = glob.glob(os.path.join(reference_dir, f'*.{args.gps_format}'))
        qry_gps_files = glob.glob(os.path.join(query_dir,     f'*.{args.gps_format}'))
        if not ref_gps_files or not qry_gps_files:
            raise FileNotFoundError("Could not find GPS files; check --gps_format and traverse dirs.")

        x_query = get_gps(qry_gps_files[0]).astype(float)      # shape: [Nq, 3] (lat, lon, t)
        x_ref   = get_gps(ref_gps_files[0]).astype(float)      # shape: [Nr, 3]

        # 1) Spatial NN with sliding window, ORIGINAL DIRECTION: query -> reference
        match_q_to_r = []
        for i, (latlon, _) in enumerate(zip(x_query[:, :2], x_query[:, 2])):
            if len(match_q_to_r) < 6:
                lo, hi = 0, int(0.25 * len(x_ref))
            elif i > 0.5 * len(x_query):
                lo, hi = match_q_to_r[-5], len(x_ref)
            else:
                lo, hi = match_q_to_r[-5], int(0.75 * len(x_ref))
            j_rel = (np.linalg.norm(x_ref[lo:hi, :2] - latlon, axis=1)).argmin()
            match_q_to_r.append(lo + j_rel)
        match_q_to_r = np.asarray(match_q_to_r, dtype=int)

        # 2) Clock to global time:
        #    Prefer known per-video offsets if available; else estimate via median.
        t_q_rel = x_query[:, 2]
        t_r_rel = x_ref[match_q_to_r, 2]

        try:
            # If your mapping is valid for this dataset, this exactly matches the notebook behavior.
            off_q = float(video_beginning[traverse_to_name[args.query]])
            off_r = float(video_beginning[traverse_to_name[args.reference]])
        except Exception:
            # Fallback: infer a constant offset that maps reference->query clock
            # (so that t_q_rel ≈ t_r_rel + off_r2q)
            off_r2q = float(np.median(t_q_rel - t_r_rel))
            off_q, off_r = 0.0, off_r2q  # apply to reference side below

        # Canonical global timelines (query and reference)
        # If using video_beginning: global = rel + per-video offset
        # If using fallback:       query stays as-is; reference is shifted by off_r (r->q)
        timestamps_canon_query = t_q_rel + (off_q if 'off_q' in locals() and off_q != 0.0 else 0.0)
        timestamps_canon_ref   = t_r_rel + (off_r if 'off_q' not in locals() or off_q == 0.0 else off_r)

        images_all_combined_set1 = {}
        images_all_combined_set2 = {}
        matches_fixedlength_combined1 = {}
        matches_fixedlength_combined2 = {}

        # Project every variant onto the canonical GPS timelines (dir-keyed, like your time-based branch)
        for q_dir, r_dir in zip(query_window_dirs, reference_window_dirs):
            ts_q = _abs_times_for_dir(q_dir)
            ts_r = _abs_times_for_dir(r_dir)

            matches_fixedlength_combined1[q_dir] = get_timestamp_matches(ts_q, timestamps_canon_query)
            matches_fixedlength_combined2[r_dir] = get_timestamp_matches(ts_r, timestamps_canon_ref)

            iq, ir = get_image_sets_on_demand(
                query_paths[q_dir], reference_paths[r_dir],
                matches_fixedlength_combined1[q_dir], matches_fixedlength_combined2[r_dir]
            )
            images_all_combined_set1[q_dir] = iq
            images_all_combined_set2[r_dir] = ir

        for q_dir, r_dir in zip(query_count_dirs, reference_count_dirs):
            ts_q = _abs_times_for_dir(q_dir)
            ts_r = _abs_times_for_dir(r_dir)

            matches_fixedlength_combined1[q_dir] = get_timestamp_matches(ts_q, timestamps_canon_query)
            matches_fixedlength_combined2[r_dir] = get_timestamp_matches(ts_r, timestamps_canon_ref)

            iq, ir = get_image_sets_on_demand(
                query_paths[q_dir], reference_paths[r_dir],
                matches_fixedlength_combined1[q_dir], matches_fixedlength_combined2[r_dir]
            )
            images_all_combined_set1[q_dir] = iq
            images_all_combined_set2[r_dir] = ir
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

    # === Feature extraction using NetVLAD ===
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
    pair_list = list(zip(query_window_dirs, reference_window_dirs)) + \
                list(zip(query_count_dirs,  reference_count_dirs))

    mats = []
    for q_dir, r_dir in pair_list:
        D = dist_matrix_all_combined.get((q_dir, r_dir))
        if D is not None and D.size:
            mats.append(D)

    if not mats:
        raise RuntimeError("No distance matrices available to ensemble.")

    max_q = max(M.shape[0] for M in mats)
    max_r = max(M.shape[1] for M in mats)

    stack = np.full((len(mats), max_q, max_r), np.nan, dtype=np.float32)
    for i, M in enumerate(mats):
        h, w = M.shape
        stack[i, :h, :w] = M

    return np.nanmean(stack, axis=0).astype(np.float32)
    
if __name__ == "__main__":
    main()