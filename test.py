import json
from collections import Counter

# JSON 파일 경로 (수정 필요)
json_path = "/home/cnrl/Workspace/ND/label.json"

# JSON 파일 불러오기
with open(json_path, 'r') as f:
    data = json.load(f)

# 레이블만 추출
labels = [item["label"] for item in data]

# 레이블별 개수 세기
label_counts = Counter(labels)

# 출력
for label, count in sorted(label_counts.items()):
    print(f"Label {label}: {count}개")