from typing import Dict, Iterable, List, Optional, Tuple

def split_aligned_data(
      aligned_data: List[Dict[str, object]],
      subj_list: Iterable[int],
      train_occurrence_max: int = 2,
  ) -> Tuple[
      List[Tuple[int, Dict[str, Optional[int]]]],
      List[Tuple[int, Dict[str, Optional[int]]]],
  ]:
      """
      aligned_data: align_subject_trials の結果（各要素に cocoidx / occurrence / subjXX フィールドがある）
      subj_list: 例 [1, 2, 3] のような被験者 ID リスト
      戻り値: (train_data, test_data)
        - 各 data[i] は (cocoidx, {"subj01": global_trial or None, ...})
      """
      train_data: List[Tuple[int, Dict[str, Optional[int]]]] = []
      test_data: List[Tuple[int, Dict[str, Optional[int]]]] = []

      for data in aligned_data:
          cocoidx = int(data["cocoidx"])
          global_trials: Dict[str, Optional[int]] = {}

          for subj in subj_list:
              s = f"subj{int(subj):02d}"
              subject_info = data.get(s)
              global_trials[s] = (
                  subject_info["global_trial"] if subject_info is not None else None
              )

          sample = (cocoidx, global_trials)
          occurrence = int(data["occurrence"])

          if occurrence <= train_occurrence_max:
              train_data.append(sample)
          else:
              test_data.append(sample)

      return train_data, test_data