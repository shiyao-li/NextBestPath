from ..utility.scone_utils import *

dir_path = os.path.abspath(os.path.dirname(__file__))
data_path = os.path.join(dir_path, "../../data/ShapeNetCore.v1")
results_dir = os.path.join(dir_path, "../../results/shapenet_reconstruction")
configs_dir = os.path.join(dir_path, "../../configs/scone/coverage_gain")

def test_loop(params, dataloader,
              scone_occ, scone_vis,
              device,
              pc_size=1024):

    # Begin test process
    coverage_dict = {}
    sum_coverages = torch.zeros(params.n_view_max, device=device)

    t0 = time.time()
    torch.cuda.empty_cache()

    print("Beginning evaluation on test dataset...")

    base_harmonics, h_polar, h_azim = get_all_harmonics_under_degree(params.harmonic_degree,
                                                                     params.view_state_n_elev,
                                                                     params.view_state_n_azim,
                                                                     device)

    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    t0 = time.time()
    computation_time = 0.

    for batch, (mesh_dict) in enumerate(dataloader):
        paths = mesh_dict['path']
        batch_size = len(paths)

        for i in range(batch_size):
            # ----------Load input mesh and ground truth data-----------------------------------------------------------

            path_i = paths[i]

            coverage_dict[path_i] = []
            coverages = torch.zeros(params.n_view_max, device=device)

            # Loading info about partial point clouds and coverages
            part_pc, coverage_matrix = get_gt_partial_point_clouds(path=path_i,
                                                                   normalization_factor=1.,
                                                                   device=device)

            # Initial dense sampling
            X_world = sample_X_in_box(x_range=params.gt_max_diagonal, n_sample=params.n_proxy_points, device=device)

            # ----------Set camera candidates for coverage prediction---------------------------------------------------
            X_cam_world, camera_dist, camera_elev, camera_azim = get_cameras_on_sphere(params, device,
                                                                                       pole_cameras=params.pole_cameras)
            n_view = 1
            view_idx = torch.randperm(len(camera_elev), device=device)[:n_view]

            prediction_cam_idx = view_idx[0]
            prediction_box_center = torch.Tensor([0., 0., params.camera_dist]).to(device)

            # Move camera coordinates from world space to prediction view space, and normalize them for prediction box
            prediction_R, prediction_T = look_at_view_transform(dist=camera_dist[prediction_cam_idx],
                                                                elev=camera_elev[prediction_cam_idx],
                                                                azim=camera_azim[prediction_cam_idx],
                                                                device=device)
            prediction_camera = FoVPerspectiveCameras(device=device, R=prediction_R, T=prediction_T)
            prediction_view_transform = prediction_camera.get_world_to_view_transform()

            X_cam = prediction_view_transform.transform_points(X_cam_world)
            X_cam = normalize_points_in_prediction_box(points=X_cam,
                                                       prediction_box_center=prediction_box_center,
                                                       prediction_box_diag=params.gt_max_diagonal)
            _, elev_cam, azim_cam = get_spherical_coords(X_cam)

            X_view = X_cam[view_idx]
            # X_cam = X_cam.view(1, params.n_camera, 3)

            # Compute initial coverage
            coverage = compute_surface_coverage_from_cam_idx(coverage_matrix, view_idx).detach().item()

            coverage_dict[path_i].append(coverage)
            coverages[0] += coverage

            # Sample random proxy points in space
            X_idx = torch.randperm(len(X_world))[:params.n_proxy_points]
            X_world = X_world[X_idx]

            for j_view in range(1, params.n_view_max):
                features = None
                args = None
                computation_t0 = time.time()

                # ----------Capture initial observations----------------------------------------------------------------

                # Points observed in initial views
                pc = torch.vstack([part_pc[pc_idx] for pc_idx in view_idx])

                # Downsampling partial point cloud
                # pc = pc[torch.randperm(len(pc))[:n_view * params.seq_len]]
                pc = pc[torch.randperm(len(pc))[:n_view * pc_size]]

                # Move partial point cloud from world space to prediction view space,
                # and normalize them in prediction box
                pc = prediction_view_transform.transform_points(pc)
                pc = normalize_points_in_prediction_box(points=pc,
                                                        prediction_box_center=prediction_box_center,
                                                        prediction_box_diag=params.gt_max_diagonal).view(1, -1, 3)

                # Move proxy points from world space to prediction view space, and normalize them in prediction box
                X = prediction_view_transform.transform_points(X_world)
                X = normalize_points_in_prediction_box(points=X,
                                                       prediction_box_center=prediction_box_center,
                                                       prediction_box_diag=params.gt_max_diagonal
                                                       )

                # Filter Proxy Points using pc shape from view cameras
                R_view, T_view = look_at_view_transform(eye=X_view,
                                                        at=torch.zeros_like(X_view),
                                                        device=device)
                view_cameras = FoVPerspectiveCameras(R=R_view, T=T_view, zfar=1000, device=device)
                X, _ = filter_proxy_points(view_cameras, X, pc.view(-1, 3), filter_tol=params.filter_tol)
                X = X.view(1, X.shape[0], 3)

                # Compute view state vector and corresponding view harmonics
                view_state = compute_view_state(X, X_view,
                                                params.view_state_n_elev, params.view_state_n_azim)
                view_harmonics = compute_view_harmonics(view_state,
                                                        base_harmonics, h_polar, h_azim,
                                                        params.view_state_n_elev, params.view_state_n_azim)
                occ_view_harmonics = 0. + view_harmonics
                if params.occ_no_view_harmonics:
                    occ_view_harmonics *= 0.
                if params.no_view_harmonics:
                    view_harmonics *= 0.

                # Compute occupancy probabilities
                with torch.no_grad():
                    occ_prob_i = compute_occupancy_probability(scone_occ=scone_occ,
                                                               pc=pc,
                                                               X=X,
                                                               view_harmonics=occ_view_harmonics,
                                                               max_points_per_pass=params.max_points_per_scone_occ_pass
                                                               ).view(-1, 1)

                proxy_points, view_harmonics, sample_idx = sample_proxy_points(X[0], occ_prob_i,
                                                                               view_harmonics.squeeze(dim=0),
                                                                               n_sample=params.seq_len,
                                                                               min_occ=params.min_occ_for_proxy_points,
                                                                               use_occ_to_sample=params.use_occ_to_sample_proxy_points,
                                                                               return_index=True)

                proxy_points = torch.unsqueeze(proxy_points, dim=0)
                view_harmonics = torch.unsqueeze(view_harmonics, dim=0)

                # ----------Predict Coverage Gains------------------------------------------------------------------------------
                visibility_gain_harmonics = scone_vis(proxy_points, view_harmonics=view_harmonics)
                if params.true_monte_carlo_sampling:
                    proxy_points = torch.unsqueeze(proxy_points[0][sample_idx], dim=0)
                    visibility_gain_harmonics = torch.unsqueeze(visibility_gain_harmonics[0][sample_idx], dim=0)

                if params.ddp or params.jz:
                    cov_pred_i = scone_vis.module.compute_coverage_gain(proxy_points,
                                                                        visibility_gain_harmonics,
                                                                        X_cam)
                else:
                    cov_pred_i = scone_vis.compute_coverage_gain(proxy_points,
                                                                 visibility_gain_harmonics,
                                                                 X_cam.view(1, -1, 3)).view(-1, 1)

                # Identify maximum gain to get NBV camera
                (max_gain, max_idx) = torch.max(cov_pred_i, dim=0)

                computation_time += time.time() - computation_t0

                # Set NBV camera parameters
                view_idx = torch.cat((view_idx, torch.Tensor([max_idx]).long().to(device)), dim=0)
                n_view += 1
                X_nbv = X_cam[max_idx:max_idx + 1]
                X_nbv_world = X_cam_world[max_idx:max_idx + 1]
                r_nbv_world = camera_dist[max_idx:max_idx + 1]
                elev_nbv_world = camera_elev[max_idx:max_idx + 1]
                azim_nbv_world = camera_azim[max_idx:max_idx + 1]

                nbv_dist = torch.Tensor([params.camera_dist]).to(device)

                nbv_R, nbv_T = look_at_view_transform(dist=r_nbv_world,
                                                      elev=elev_nbv_world,
                                                      azim=azim_nbv_world,
                                                      device=device)

                R_view = torch.vstack((R_view, nbv_R))
                # screen_dist = torch.vstack((screen_dist, nbv_dist))
                X_view = torch.vstack((X_view, X_nbv))

                # Computing surface coverage
                coverage = compute_surface_coverage_from_cam_idx(coverage_matrix, view_idx).detach().item()
                # avg_coverage += coverage / batch_size
                coverage_dict[path_i].append(coverage)
                coverages[j_view] += coverage

            sum_coverages += coverages

        # ----------Metrics computation on batch----------

        if batch % 10 == 0:
            torch.cuda.empty_cache()
            print("--- Batch", batch, "---")
            print("Batch size:", batch_size)
            # print("Coverage:", avg_coverage / (batch + 1))
            print("Coverages:", sum_coverages / ((batch + 1) * params.batch_size))
            print("Nb of meshes done:", (batch + 1) * params.batch_size)
            print("Computation time:", computation_time, '\n')

    results = {}
    # results["occ_threshold"] = occ_threshold
    # results["uncertainty_threshold"] = params.uncertainty_threshold
    # results["uncertainty_mode"] = params.uncertainty_mode
    # results["compute_cross_correction"] = params.compute_cross_correction
    # results["nbv_mode"] = nbv_mode
    results["coverages"] = coverage_dict

    # print("Results:", results)

    print("Avg coverages loss:", sum_coverages.detach().cpu() / len(dataloader.dataset))
    print("Done in", (time.time() - t0) / 3600., "hours!")
    print("Computation time:", computation_time)
    print("Average computation time:", computation_time / len(dataloader.dataset))

    print("Terminated in", (time.time() - t0) / 60., "minutes.")
    return results


def run_test(test_params):
    params_path = os.path.join(configs_dir, test_params.params_name)
    results_json_path = os.path.join(results_dir, test_params.results_json_name)

    params = load_params(params_path)

    params.data_path = test_params.data_path

    params.scone_occ_model_name = test_params.scone_occ_model_name
    params.scone_vis_model_name = test_params.scone_vis_model_name
    params.pc_size = test_params.pc_size
    params.n_view_max = test_params.n_view_max
    params.test_novel = test_params.test_novel
    params.test_number = test_params.test_number

    params.random_seed = test_params.random_seed
    params.torch_seed = test_params.torch_seed

    params.jz = False
    params.numGPU = test_params.numGPU
    params.batch_size = 4
    params.WORLD_SIZE = 1

    # Set device
    device = setup_device(params, ddp_rank=None)

    # Load models
    print("Loading SconeOcc...")
    scone_occ = load_scone_occ(params, params.scone_occ_model_name, ddp_model=True, device=device)
    print("Occ model name:", params.scone_occ_model_name)
    print("Model has", count_parameters(scone_occ) / 1e6, "M parameters.")
    scone_occ.eval()

    print("Loading SconeVis...")
    scone_vis = load_scone_vis(params, params.scone_vis_model_name, ddp_model=True, device=device)
    print("Vis model name:", params.scone_vis_model_name)
    print("Model has", count_parameters(scone_vis) / 1e6, "M parameters.")
    scone_vis.eval()

    if params.test_novel:
        print("Test on novel categories.")

    train_dataloader, val_dataloader, test_dataloader = get_shapenet_dataloader(batch_size=params.batch_size,
                                                                                ddp=params.ddp, jz=params.jz,
                                                                                world_size=None, ddp_rank=None,
                                                                                test_number=params.test_number,
                                                                                test_novel=params.test_novel,
                                                                                load_obj=False,
                                                                                data_path=params.data_path)

    # Main loop
    eval_results = []
    eval_results.append(test_loop(params, test_dataloader, scone_occ, scone_vis, device, pc_size=params.pc_size))

    for res in eval_results:
        for key in res:
            if type(res[key]) == torch.Tensor:
                res[key] = res[key].detach().item()

            if type(res[key]) == np.ndarray:
                res[key] = float(res[key])

    with open(results_json_path, 'w') as outfile:
        json.dump(eval_results, outfile)
    print("Saved data about test losses in", results_json_path)
