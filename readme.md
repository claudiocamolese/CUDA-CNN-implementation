Dopo averlo salvato, nella cartella `build` puoi fare:

```bash
cmake ..
cmake --build . --targ et naive_cnn
cmake --build . --target optimized_cnn
```

```bash
srun --gres=gpu:1 --cpus-per-task=4 --mem=4GB ./naive_cnn
srun --gres=gpu:1 --cpus-per-task=4 --mem=4GB ./optimized_cnn
```