result_brain_dict = collate_result_dict_list(result_brain_dict_list)
result_recon_dict = collate_result_dict_list(result_recon_dict_list)
result_label_dict = collate_result_dict_list(result_label_dict_list)

pattern_corr_result = []
profile_corr_result = []

for s_t, s_s in permutations(subject_list, 2):
    target_brain = result_brain_dict[f"x{s_t}"]
    target_label = result_label_dict[f"x{s_t}"]
    recon_brain = result_recon_dict[f"cross_recon_x{s_t}__x{s_s}"]
    recon_label = result_label_dict[f"x{s_s}"]

    label_set = get_label_set([f"{s_t}", f"{s_s}"], result_label_dict, set_mode="and")

    pattern_corr_list_dict = calc_pattern_corr_list_dict(target_brain, recon_brain, target_label, recon_label, label_set)
    for image_idx, corr_list in pattern_corr_list_dict.items():
        for corr in corr_list:
            pattern_corr_result.append({
                'Subject_target': s_t, 
                'Subject_source': s_s,           
                'Correlation': corr.cpu().numpy(), 
                'Image_idx': image_idx}
            )

    # profile_corrs = calc_profile_correlation(target_brain, recon_brain, target_label, recon_label, label_set)
    # for i, corr in enumerate(profile_corr):
    #     profile_corr_result.append({
    #         'Subject_target': s_t, 
    #         'Subject_source': s_s,           
    #         'Correlation': corr.cpu().numpy(), 
    #         'Voxel_idx': i}
    #     )

if group_train:
    if group_test:
        save_result(pattern_corr_result, output_dir+"train_grouped/test_grouped/", f"pattern_correlation_dmvae_subj{''.join(map(str, subject_list))}.csv")
        # save_result(profile_corr_result, output_dir+"train_grouped/test_grouped/", f"profile_correlation_dmvae_subj{''.join(map(str, subject_list))}.csv")
    else:
        save_result(pattern_corr_result, output_dir+"train_grouped/no_test_grouped/", f"pattern_correlation_dmvae_subj{''.join(map(str, subject_list))}.csv")
        # save_result(profile_corr_result, output_dir+"train_grouped/no_test_grouped/", f"profile_correlation_dmvae_subj{''.join(map(str, subject_list))}.csv")        
        
else:
    if group_test:
        save_result(pattern_corr_result, output_dir+"no_train_grouped/test_grouped/", f"pattern_correlation_dmvae_subj{''.join(map(str, subject_list))}.csv")
        # save_result(profile_corr_result, output_dir+"no_train_grouped/test_grouped/", f"profile_correlation_dmvae_subj{''.join(map(str, subject_list))}.csv")
    else:
        save_result(pattern_corr_result, output_dir+"no_train_grouped/no_test_grouped/", f"pattern_correlation_dmvae_subj{''.join(map(str, subject_list))}.csv")
        # save_result(profile_corr_result, output_dir+"no_train_grouped/no_test_grouped/", f"profile_correlation_dmvae_subj{''.join(map(str, subject_list))}.csv")