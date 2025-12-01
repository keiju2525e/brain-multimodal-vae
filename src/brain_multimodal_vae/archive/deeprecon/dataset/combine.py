from itertools import product

def conbine_cross_occurrences(data, subj_list):
    """
    data（各要素が {'image_index': int, 'occurrence': int, 'subj01': array, ...}）から、
    同じ image_index 内で被験者ごとの occurrence を自由に組み合わせたサンプルのリストを生成する。

    返り値の各要素は:
        {
            "image_index": <int>,
            "subj01": <array>,
            "subj02": <array>,
            ...
        }

    引数:
        data: 辞書のリスト
        subj_list: 数値のリスト（例: [1,3,5] や [1,2,3,4,5]）
    """
    subject_keys = [f"subj{n:02d}" for n in subj_list]

    # image_index ごとにレコードをまとめる
    records_grouped_by_image_index = {}
    for record in data:
        image_index_value = record["image_index"]
        if image_index_value not in records_grouped_by_image_index:
            records_grouped_by_image_index[image_index_value] = []
        records_grouped_by_image_index[image_index_value].append(record)

    all_output_samples = []

    for image_index_value, records_in_group in records_grouped_by_image_index.items():
        # 被験者 -> { occurrence: array } のマップを構築
        subject_to_occurrence_map = {subject_key: {} for subject_key in subject_keys}
        for record in records_in_group:
            occurrence_value = record["occurrence"]
            for subject_key in subject_keys:
                if subject_key in record:
                    subject_to_occurrence_map[subject_key][occurrence_value] = record[subject_key]

        # 各被験者の occurrence 候補（ソートして安定化）
        ordered_subject_keys = list(subject_keys)
        occurrence_choice_lists = [sorted(subject_to_occurrence_map[subj].keys()) for subj in ordered_subject_keys]

        # 全組み合わせを生成（欠損がある組み合わせはスキップ）
        for occurrence_choice_tuple in product(*occurrence_choice_lists):
            output_sample = {"image_index": image_index_value}
            valid_combination = True
            for subject_key, chosen_occurrence in zip(ordered_subject_keys, occurrence_choice_tuple):
                if chosen_occurrence in subject_to_occurrence_map[subject_key]:
                    output_sample[subject_key] = subject_to_occurrence_map[subject_key][chosen_occurrence]
                else:
                    valid_combination = False
                    break
            if valid_combination:
                all_output_samples.append(output_sample)

    return all_output_samples
