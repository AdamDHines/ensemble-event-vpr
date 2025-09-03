'''
Imports
'''
import os, argparse, glob, torch

import numpy as np
import code_helpers_public as chp 

from tqdm import tqdm
from read_gps import get_gps
from correspondence_event_camera_frame_camera import traverse_to_name, video_beginning

def main():
    parser = argparse.ArgumentParser()

    # Input arguments
    parser.add_argument('--dataset', '-ds', type=str, help='Dataset name',
                        required=True)
    parser.add_argument('--reference', '-r', type=str, help='Reference traverse name',
                        required=True)
    parser.add_argument('--query', '-q', type=str, help='Query traverse name',
                        required=True)
    parser.add_argument('--window_duration', '-w', action=chp.ListOrLiteral, nargs='+', help='Fixed window duration in ms',
                        default=[250, 500, 750, 1000])
    parser.add_argument('--num_events_per_pixel', '-n', action=chp.ListOrLiteral, nargs='+', help='Number of events per pixel',
                        default=[0.2, 0.4, 0.6, 0.8])
    parser.add_argument('--gps_available', action='store_true', help='Whether GPS data is available',
                        default=False)
    parser.add_argument('--gps_format', type=str, help='GPS format')
    
    # Directories
    parser.add_argument('--dataset_folder', '-d', type=str, help='Path to dataset folder',
                        required=True)
    parser.add_argument('--netvlad_folder', '-nv', type=str, help='Path to NetVLAD folder',
                        required=True)
    parser.add_argument('--frames_subfolder', type=str, help='Subfolder name for frames',
                        default='reconstruction')
    parser.add_argument('--outdir', '-o', type=str, help='Output directory for results',
                        required=True)
    
    # Parse the arguments
    args = parser.parse_args()

    # Replace all "." with. "_" for num_events_per_pixel
    args.num_events_per_pixel = [str(x).replace(".", "_") for x in args.num_events_per_pixel]

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
        reference_paths[subfolder] = chp.get_image_paths(subfolder)

    for subfolder in query_combined_dirs:
        query_paths[subfolder] = chp.get_image_paths(subfolder)
    
    # Initialize dictionaries to hold the final aligned and lazy-loaded image sets
    images_all_combined_set1 = {}
    images_all_combined_set2 = {}

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
    
    # Both paths (GPS and non-GPS) now converge here to create the final timeline.
    if not args.window_duration:
         raise ValueError("Alignment requires at least one --window_duration to be specified as an anchor.")
    anchor_param = args.window_duration[0]
    q_anchor_dir = os.path.join(f'{args.dataset_folder}',f'{args.dataset}',f'{args.query}',f'{args.query}-{args.frames_subfolder}-{anchor_param}',f'{args.frames_subfolder}')
    r_anchor_dir = os.path.join(f'{args.dataset_folder}',f'{args.dataset}',f'{args.reference}',f'{args.reference}-{args.frames_subfolder}-{anchor_param}',f'{args.frames_subfolder}')

    if not os.path.isdir(q_anchor_dir) or not os.path.isdir(r_anchor_dir):
        raise FileNotFoundError(f"Anchor directories not found for alignment: {q_anchor_dir} | {r_anchor_dir}")

    print(f"Using reconstruction parameter '{anchor_param}' as the time anchor.")
    
    # 2. Get the dense, absolute timestamps from the anchor image sets
    q_times_anchor = chp._abs_times_for_dir(q_anchor_dir)
    r_times_anchor = chp._abs_times_for_dir(r_anchor_dir)

    # 3. Apply the pre-calculated time offsets (they are 0.0 if GPS was not used)
    q_times_synced = q_times_anchor + off_q
    r_times_synced = r_times_anchor + off_r
    
    # 4. Use the robust _pair_canonical function on the now-synchronized dense timelines
    max_dt = getattr(args, 'max_dt', 0.06)
    timestamps_canon_query, timestamps_canon_ref = chp._pair_canonical(q_times_synced, r_times_synced, max_dt)
    
    if len(timestamps_canon_query) == 0:
        raise RuntimeError("Anchor alignment failed. No matching timestamps found. Check data or increase max_dt.")

    print(f"Projecting all data variants onto the canonical timeline of length {len(timestamps_canon_query)}...")

    all_query_dirs = query_window_dirs + query_count_dirs
    all_ref_dirs = reference_window_dirs + reference_count_dirs

    for q_dir, r_dir in zip(all_query_dirs, all_ref_dirs):
        ts_q = chp._abs_times_for_dir(q_dir) + off_q # Apply offsets here too for consistency
        ts_r = chp._abs_times_for_dir(r_dir) + off_r

        matches_q = chp.get_timestamp_matches(ts_q, timestamps_canon_query)
        matches_r = chp.get_timestamp_matches(ts_r, timestamps_canon_ref)

        iq, ir = chp.get_image_sets_on_demand(
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

    def _safe_pair_dir(q_dir: str, r_dir: str) -> str:
        q_rel = os.path.relpath(q_dir, args.dataset_folder).replace(os.sep, "__")
        r_rel = os.path.relpath(r_dir, args.dataset_folder).replace(os.sep, "__")
        return f"{q_rel}__VS__{r_rel}", os.path.join(overall_data_dir, "pairs", f"{q_rel}__VS__{r_rel}")

    ensemble_matrix = None
    num_matrices_in_ensemble = 0

    # Combine all pairs into a single list for a unified progress bar
    all_pairs = list(zip(query_window_dirs, reference_window_dirs)) + \
                list(zip(query_count_dirs,  reference_count_dirs))

    print(f"Starting combined feature extraction and ensembling for {len(all_pairs)} pairs...")
    # Define the model
    num_clusters=64
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model, _ = chp.build_model(num_clusters=num_clusters, device=device, add_wpca=True, netvlad_folder=args.netvlad_folder)

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
            unique_id, pair_dir = _safe_pair_dir(q_dir, r_dir)
            os.makedirs(pair_dir, exist_ok=True)
            f1_path = os.path.join(pair_dir, "netvlad_features_all_set1.npy")
            f2_path = os.path.join(pair_dir, "netvlad_features_all_set2.npy")
            
            features_q = chp.get_vlad_features(
                                            model,
                                            images_set=imgs_q,        
                                            save_name=f1_path,             
                                            batch_size=16,
                                            num_workers=4,
                                            tqdm_position=1,
                                            device=None,                
                                            target_size=(480, 640),         
                                            mmap_safely=True,                 
                                        )
            features_r = chp.get_vlad_features(
                                            model,
                                            images_set=imgs_r,                 
                                            save_name=f2_path,                 
                                            batch_size=16,
                                            num_workers=4,
                                            tqdm_position=1,
                                            device=None,                   
                                            target_size=(480, 640),          
                                            mmap_safely=True,                  
                                        )

            tq = torch.as_tensor(features_q, device=device, dtype=torch.float32)
            tr = torch.as_tensor(features_r, device=device, dtype=torch.float32)

            D = (1 - (tq @ tr.T)).T.cpu().numpy()   # shape (Nr, Nq) as before

            # Save the intemerdiate distance matrix for this pair with unique name
            D_path = os.path.join(args.outdir, f'{unique_id}.npy') 
            np.save(D_path, D)

            del tq, tr
            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps":
                torch.mps.empty_cache()
                
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

    # Save the final ensemble matrix
    os.makedirs(args.outdir, exist_ok=True)
    ensemble_path = os.path.join(args.outdir, f"ensemble_distance_{comparison_id}.npy")
    np.save(ensemble_path, final_ensemble_matrix)
    print(f"Ensemble distance matrix saved to: {ensemble_path}")
    
if __name__ == "__main__":
    main()