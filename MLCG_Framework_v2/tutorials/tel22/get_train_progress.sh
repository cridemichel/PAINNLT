tail -n 10 train_v1.log | grep -Eo 'Epoca: [0-9]+ | Train Loss: [0-9.]+ | Val Loss: [0-9.]+' || true
