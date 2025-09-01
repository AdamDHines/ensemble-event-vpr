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
                        default=[66, 88, 120, 140])
    parser.add_argument('--num_events_per_pixel', '-n', type=list, help='Number of events per pixel',
                        default=[66, 88, 120, 140])
    parser.add_argument('--gps_available', action='store_true', help='Whether GPS data is available',
                        default=False)
    parser.add_argument('--gps_format', type=str, help='GPS format (nmea or kml)',
                        default='nmea')
    
    # Directories
    parser.add_argument('--dataset_folder', '-d', type=str, help='Path to dataset folder',
                        default="/media/adam/vprdatasets/data/event-datasets")
    parser.add_argument('--netvlad_folder', '-nv', type=str, help='Path to NetVLAD folder',
                        default="/home/adam/repo/netvlad_tf_open")
    parser.add_argument('--frames_subfolder', type=str, help='Subfolder name for frames',
                        default='reconstruction')
    
    # Parse the arguments
    args = parser.parse_args()
    args.gps_available = True
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
    
    # Initialize dictionaries to hold the final aligned and lazy-loaded image sets
    images_all_combined_set1 = {}
    images_all_combined_set2 = {}
    matches_fixedlength_combined1 = {}
    matches_fixedlength_combined2 = {}

    # Initialize time offsets. These will be calculated if GPS is available.
    off_q, off_r = 0.0, 0.0

    if args.gps_available:
        # --- STRATEGY 1: Determine Global Time Offset using GPS ---
        print("Performing GPS-based time offset calculation...")

        # 1. Load GPS files
        reference_dir = os.path.join(args.dataset_folder, args.dataset, args.reference)
        query_dir     = os.path.join(args.dataset_folder, args.dataset, args.query)
        ref_gps_files = glob.glob(os.path.join(reference_dir, f'*.{args.gps_format}'))
        qry_gps_files = glob.glob(os.path.join(query_dir,     f'*.{args.gps_format}'))
        
        if not ref_gps_files or not qry_gps_files:
            raise FileNotFoundError(f"GPS files with format '.{args.gps_format}' not found.")

        x_query = get_gps(qry_gps_files[0]).astype(float)
        x_ref   = get_gps(ref_gps_files[0]).astype(float)

        # 2. Find spatial correspondence
        match_q_to_r = []
        for i, (latlon, _) in enumerate(zip(x_query[:, :2], x_query[:, 2])):
            if len(match_q_to_r) < 6: lo, hi = 0, int(0.25 * len(x_ref))
            elif i > 0.5 * len(x_query): lo, hi = match_q_to_r[-5], len(x_ref)
            else: lo, hi = match_q_to_r[-5], int(0.75 * len(x_ref))
            j_rel = (np.linalg.norm(x_ref[lo:hi, :2] - latlon, axis=1)).argmin()
            match_q_to_r.append(lo + j_rel)
        match_q_to_r = np.asarray(match_q_to_r, dtype=int)

        # 3. Calculate the precise time offsets with robust fallback
        t_q_rel = x_query[:, 2]
        t_r_rel = x_ref[match_q_to_r, 2]
        try:
            off_q = float(video_beginning[traverse_to_name[args.query]])
            off_r = float(video_beginning[traverse_to_name[args.reference]])
            print("Successfully used 'video_beginning' offsets for precise GPS time synchronization.")
        except (KeyError, AttributeError):
            print("Warning: Could not find 'video_beginning' offsets. Falling back to median time offset.")
            off_r_to_q = float(np.median(t_q_rel - t_r_rel))
            off_q = 0.0
            off_r = -off_r_to_q
    else:
        print("GPS not available. Proceeding with time-based anchor alignment.")
    
    # --- UNIFIED CANONICAL TIMELINE CREATION ---
    # Both paths (GPS and non-GPS) now converge here to create the final timeline.
    
    # 1. Select a reliable "anchor" data variant to provide the dense timestamps.
    if not args.window_duration:
         raise ValueError("Alignment requires at least one --window_duration to be specified as an anchor.")
    anchor_param = args.window_duration[0]
    q_anchor_dir = os.path.join(f'{args.dataset_folder}',f'{args.dataset}',f'{args.query}',f'{args.query}-{args.frames_subfolder}-{anchor_param}',f'{args.frames_subfolder}')
    r_anchor_dir = os.path.join(f'{args.dataset_folder}',f'{args.dataset}',f'{args.reference}',f'{args.reference}-{args.frames_subfolder}-{anchor_param}',f'{args.frames_subfolder}')

    if not os.path.isdir(q_anchor_dir) or not os.path.isdir(r_anchor_dir):
        raise FileNotFoundError(f"Anchor directories not found for alignment: {q_anchor_dir} | {r_anchor_dir}")

    print(f"Using reconstruction parameter '{anchor_param}' as the time anchor.")
    
    # 2. Get the dense, absolute timestamps from the anchor image sets
    q_times_anchor = _abs_times_for_dir(q_anchor_dir)
    r_times_anchor = _abs_times_for_dir(r_anchor_dir)

    # 3. Apply the pre-calculated time offsets (they are 0.0 if GPS was not used)
    q_times_synced = q_times_anchor + off_q
    r_times_synced = r_times_anchor + off_r
    
    # 4. Use the robust _pair_canonical function on the now-synchronized dense timelines
    max_dt = getattr(args, 'max_dt', 0.06)
    timestamps_canon_query, timestamps_canon_ref = _pair_canonical(q_times_synced, r_times_synced, max_dt)
    
    if len(timestamps_canon_query) == 0:
        raise RuntimeError("Anchor alignment failed. No matching timestamps found. Check data or increase max_dt.")

    # =================================================================================
    # ### GLOBAL PROJECTION STEP (Now Correct for Both Paths) ###
    # =================================================================================
    print(f"Projecting all data variants onto the canonical timeline of length {len(timestamps_canon_query)}...")

    all_query_dirs = query_window_dirs + query_count_dirs
    all_ref_dirs = reference_window_dirs + reference_count_dirs

    for q_dir, r_dir in zip(all_query_dirs, all_ref_dirs):
        ts_q = _abs_times_for_dir(q_dir) + off_q # Apply offsets here too for consistency
        ts_r = _abs_times_for_dir(r_dir) + off_r

        matches_q = get_timestamp_matches(ts_q, timestamps_canon_query)
        matches_r = get_timestamp_matches(ts_r, timestamps_canon_ref)

        iq, ir = get_image_sets_on_demand(
            query_paths[q_dir],
            reference_paths[r_dir],
            matches_q,
            matches_r
        )
        images_all_combined_set1[q_dir] = iq
        images_all_combined_set2[r_dir] = ir

    print("Global alignment and data projection complete.")

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
    config = tf.compat.v1.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1, gpu_options=gpu_options, allow_soft_placement=True)
    config.log_device_placement = True
    TF_ENABLE_GARBAGE_COLLECTION = True
    sess = tf.compat.v1.Session(config=config)
    saver.restore(sess, nets.defaultCheckpoint())
    print("NetVLAD ready")

    def _safe_pair_dir(q_dir: str, r_dir: str) -> str:
        q_rel = os.path.relpath(q_dir, args.dataset_folder).replace(os.sep, "__")
        r_rel = os.path.relpath(r_dir, args.dataset_folder).replace(os.sep, "__")
        return os.path.join(overall_data_dir, "pairs", f"{q_rel}__VS__{r_rel}")

    ensemble_matrix = None
    num_matrices_in_ensemble = 0

    # Combine all pairs into a single list for a unified progress bar
    all_pairs = list(zip(query_window_dirs, reference_window_dirs)) + \
                list(zip(query_count_dirs,  reference_count_dirs))

    print(f"Starting combined feature extraction and ensembling for {len(all_pairs)} pairs...")
    
    with tqdm(total=len(all_pairs), position=0, leave=True, desc="Processing Pairs") as pbar:
        for q_dir, r_dir in all_pairs:
            # --- 1. Get image sets for the current pair ---
            imgs_q = images_all_combined_set1.get(q_dir)
            imgs_r = images_all_combined_set2.get(r_dir)

            # Handle cases where alignment produced no images
            if imgs_q is None or imgs_r is None or len(imgs_q) == 0 or len(imgs_r) == 0:
                pbar.update(1)
                pbar.set_postfix_str(f"Skipping empty pair: {os.path.basename(q_dir)}")
                continue

            # --- 2. Extract features for the current pair ---
            pair_dir = _safe_pair_dir(q_dir, r_dir)
            os.makedirs(pair_dir, exist_ok=True)
            f1_path = os.path.join(pair_dir, "netvlad_features_all_set1.npy")
            f2_path = os.path.join(pair_dir, "netvlad_features_all_set2.npy")
            
            features_q = get_vlad_features(sess, net_out, image_batch, imgs_q, f1_path)
            features_r = get_vlad_features(sess, net_out, image_batch, imgs_r, f2_path)

            # --- 3. Compute the temporary distance matrix ---
            # This matrix ('D') exists only within this loop iteration.
            D = scipy.spatial.distance.cdist(
                features_q, features_r, metric="cosine"
            ).T

            # Sanity check the shape
            assert D.shape == (len(imgs_r), len(imgs_q)), \
                f"Distance shape mismatch for pair {os.path.basename(r_dir)} vs {os.path.basename(q_dir)}"

            # --- 4. Add to the ensemble accumulator in-place ---
            if ensemble_matrix is None:
                # First valid matrix: initialize the accumulator.
                # Use float64 for summation to avoid precision loss.
                pbar.set_postfix_str(f"Initializing ensemble with shape {D.shape}")
                ensemble_matrix = D.astype(np.float64)
                num_matrices_in_ensemble = 1
            else:
                # Subsequent matrix: add it directly to the existing accumulator.
                # The global alignment guarantees all D matrices have the same shape.
                assert ensemble_matrix.shape == D.shape, \
                    "Shape mismatch during ensembling! This should not happen with unified global alignment."
                ensemble_matrix += D
                num_matrices_in_ensemble += 1
                pbar.set_postfix_str(f"Added matrix #{num_matrices_in_ensemble} to ensemble")
            
            pbar.update(1)

    # --- 5. Finalize the ensemble ---
    # After the loop, normalize the summed matrix to get the mean.
    if ensemble_matrix is None:
        raise RuntimeError("No valid distance matrices were found to create an ensemble.")

    print(f"\nFinalizing ensemble from {num_matrices_in_ensemble} matrices...")
    ensemble_matrix /= num_matrices_in_ensemble

    # Cast back to standard float32 for the final output
    final_ensemble_matrix = ensemble_matrix.astype(np.float32)
    import matplotlib.pyplot as plt
    plt.imshow(final_ensemble_matrix)
    plt.colorbar()
    plt.show()
    return final_ensemble_matrix
    
if __name__ == "__main__":
    main()