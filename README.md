
## Setup
```
conda env create -f environment.yml
conda activate DuCAR
```

## Evaluation

### POPE

- Generate the LVLM's responsed and save them:

```bash
python pope_eval.py --model model_name --batch_size 1  --data-path /path/to/COCO --pope-type random --use-attn --alpha 0.6 --beta 0.2 --use-cfg --gamma 1.1 --start-layer 2 --end-layer 18 --threshold_p 0.85
```

- Calculate POPE using the answer file:

```bash
python pope_ans.py --ans_file /path/to/answer.json
```

### CHAIR

- Generate the LVLM's responses and save them in a jsonl file:

```bash
python modPAI/pope_eval.py --model model_name --batch_size 1  --data-path /path/to/COCO --pope-type random --use-attn --alpha 0.6 --beta 0.2 --use-cfg --gamma 1.1 --start-layer 2 --end-layer 18 --threshold_p 0.85
```

- Calculate CHAIR using the generated jsonl file:

```bash
python chair.py --cap_file /path/to/jsonl
```

