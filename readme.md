Dopo averlo salvato, nella cartella `build` puoi fare:

```bash
cmake ..
cmake --build . --target cnn_baseline
cmake --build . --target cnn_optimized
```

```bash
srun --gres=gpu:1 --cpus-per-task=4 --mem=4GB ./cnn_baseline
srun --gres=gpu:1 --cpus-per-task=4 --mem=4GB ./cnn_optimized
```