# Caro 9×9 và Connect Four

Bài tập lớn AI giữa kỳ 2026–2027. Caro có Random, MiniMax, Alpha-Beta và
MCTS + UCT. Connect Four có Random và dùng lại cùng bộ tìm kiếm MCTS.
Cả hai trò có chế độ người đấu máy và máy đấu máy.

## Cài đặt

Mở terminal trong thư mục `MaNguon`. Cần Python 3.10 trở lên.

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Chỉ cài `requirements.txt` nếu cần chơi và chạy benchmark mà không chạy pytest.
CLI của trò chơi chỉ dùng thư viện chuẩn Python.

## Chạy trò chơi

```bash
python -m src.main --help
python -m src.main --game caro --mode human-ai --agent mcts --iterations 8 --rollout heuristic
python -m src.main --mode ai-ai --agent1 minimax --agent2 mcts --depth 2 --alpha-beta --iterations 4 --quiet
python -m src.main --game connect4 --mode human-ai --agent mcts --iterations 8
python -m src.main --game connect4 --mode ai-ai --agent1 mcts --agent2 random --iterations 4 --quiet
```

- Caro nhận hàng và cột từ 1 đến 9, ví dụ `5 5`. Có từ năm quân liền nhau
  là thắng, kể cả khi bị chặn hai đầu.
- Connect Four nhận số cột từ 1 đến 7. Quân rơi xuống ô trống thấp nhất.
  Bốn quân liền nhau là thắng.
- X đi trước. Có thể dùng `--x` và `--o` để đặt tác nhân theo bên,
  chẳng hạn `--x mcts --o human`.
- `--depth` là độ sâu MiniMax. `--alpha-beta` bật cắt tỉa.
  `--no-alpha-beta` chạy MiniMax thuần. Tên tác nhân `alphabeta` bật cắt tỉa sẵn.
- `--iterations` là số mô phỏng MCTS mỗi lượt. `--c` là hệ số khám phá UCT.
  `--rollout random` chọn đều trong mọi nước hợp lệ. `--rollout heuristic`
  ưu tiên thắng ngay, chặn thắng rồi mới phát triển chuỗi.
- `--tactical` giúp MiniMax chọn nhánh thắng sớm hơn trong phạm vi tìm kiếm.
  Tại gốc cây MCTS, cờ này giới hạn tìm kiếm vào các nước thắng ngay hoặc
  nước chặn duy nhất. Nước đi cuối cùng vẫn được chọn theo số lượt thăm.
  Các benchmark kèm theo dùng cấu hình cơ sở, chưa bật `--tactical`.
- `--seed` giúp tái lập chuỗi quyết định ngẫu nhiên. Nhật ký JSONL ghi tọa độ
  nội bộ từ 0, thời gian, số nút, độ sâu hoặc số mô phỏng. Dùng `--log` để đổi tệp.

```bash
python -m src.main --game caro --mode ai-ai --agent1 alphabeta --agent2 mcts --depth 5 --iterations 1000 --tactical --quiet
```

## Chạy thực nghiệm

```bash
python -m experiments.benchmark --preset smoke
python -m experiments.benchmark --preset full --output experiments/results/reproduction
python -m experiments.compare_pruning
python -m experiments.plot_results experiments/results/verified_full/results.csv --output experiments/results/replotted
```

- `smoke` kiểm tra toàn bộ quy trình với hai ván/cấu hình và ngân sách mô
  phỏng nhỏ. `full` mặc định chạy 20 ván/cấu hình, tổng cộng 260 ván.
- Dùng `--group baseline`, `uct_c`, `iterations`, `rollout` hoặc `connect4`
  để chạy riêng một nhóm. Chọn thư mục đầu ra mới cho mỗi lần chạy.
- Hai ván trong một cặp đổi bên X/O và giữ nguyên seed của từng tác nhân.
  Đặt `--seed 2026` để dùng cùng lịch seed với kết quả kèm theo.
- Khảo sát `c` dùng 300 mô phỏng mặc định. Khảo sát iteration dùng 100,
  300 và 500 với `c=1.4` mặc định. Đối thủ mặc định trong hai khảo sát là
  Random, có thể đổi bằng `--opponent AlphaBeta`.
- Hướng A đối đầu trực tiếp heuristic và random rollout trên Caro. Mỗi bên
  có 100 mô phỏng mỗi lượt ở cấu hình mặc định, chỉnh bằng
  `--rollout-iterations`. Hướng B đối sánh MCTS với Random trên Connect Four.
- `games.csv` lưu từng ván. `results.csv` lưu tổng hợp theo cấu hình.
  `metadata.json` ghi môi trường, tham số và phương pháp đo. Thời gian
  trung bình là trung bình trên các lượt đi của tác nhân được đánh giá.
  Seed tái lập kết quả ván, còn thời gian phụ thuộc vào máy và tải hệ thống.
- `experiments/results/verified_full/` chứa lần chạy đầy đủ kèm theo.
  `pruning_comparison.csv` đối chiếu số nút, nước đi và điểm tìm kiếm ở
  ba thế cờ độ sâu 3.

Trong lần chạy kèm theo, heuristic rollout thắng 17 và thua 3 trước
random rollout ở cùng ngân sách 100 mô phỏng. MCTS thắng 20/20 trước
Random trên Connect Four. Trên Caro, MCTS thắng 20/20 trước Random ở mọi
cấu hình khảo sát nhưng thua cả 20 ván đối đầu Alpha-Beta ở độ sâu 2,
với MCTS dùng 300 mô phỏng và `c=1.4`. Các số này áp dụng cho lịch seed,
đối thủ và cấu hình ghi trong metadata.

## Đọc mã theo thứ tự

| Mô-đun | Vai trò |
| --- | --- |
| `src/game.py`, `src/connect4.py` | Luật chơi, nước hợp lệ, phát hiện thắng và hòa |
| `src/minimax.py` | Điểm theo cửa sổ 5 ô, MAX/MIN và cắt tỉa |
| `src/mcts.py` | Chọn nút, mở rộng, mô phỏng, lan truyền kết quả |
| `src/agents.py` | Random và cấu trúc thống kê chung |
| `src/main.py` | Nhập nước đi, chọn tác nhân, in bàn cờ và ghi JSONL |
| `experiments/`, `tests/` | Đối sánh, đồ thị và kiểm thử tự động |

Trong MCTS, mỗi nút lưu điểm của người vừa tạo ra nút đó. UCT ưu tiên nút
chưa thăm, sau đó cân bằng điểm trung bình và mức khám phá. Khi hết lượt
mô phỏng, chọn nút con được thăm nhiều nhất. Nếu bằng nhau thì lấy nút
được mở rộng trước. Số lượt thăm phản ánh lượng tìm kiếm đã dành cho
một nước, trong khi tỷ lệ thắng của nút ít mẫu có thể dao động mạnh.

Hướng C cần đề tài được giảng viên phê duyệt. Gói này triển khai hướng A và B.

Công cụ AI hỗ trợ rà soát và chỉnh sửa mã, bổ sung kiểm thử và chạy benchmark.
