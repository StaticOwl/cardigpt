# Cardi-GPT: An Expert ECG-Record Processing Chatbot

![IEEE SoutheastCon 2025](https://img.shields.io/badge/Published-IEEE%20SoutheastCon%202025-blue)

> 📄 **Official Publication:**  
> "Cardi-GPT: An Expert ECG-Record Processing Chatbot," *2025 IEEE SoutheastCon*, DOI: [10.1109/SoutheastCon56624.2025.10971509](https://doi.org/10.1109/SoutheastCon56624.2025.10971509)

---

# Overview

Cardi-GPT is an expert system that integrates deep learning, fuzzification, and natural language interaction to streamline 12-lead ECG interpretation and clinical communication.

This repository contains the primary codebase and associated research materials developed as part of the IEEE SoutheastCon 2025 publication.

---

# How to Run

```bash
# Predict using a pretrained model
python main.py --model_name resnet_tr --type predict --pretrained --test_model_name 54-81-0.5748.pth --test_file A0064

# Predict without pretrained weights
python main.py --model_name resnet_tr --type predict --test_model_name 54-81-0.5748.pth --test_file A0064

# Train a model from scratch (with optional pretrained base)
python main.py --model_name resnet_tr --max_epoch 100 --num_workers 10 --type train --pretrained
```

---

# Project Structure

- `/papers/` : Literature reviews and related research documentation.
- `/models/` : Saved model checkpoints.
- `/src/` : Core implementation (prediction, training, fuzzification, chatbot integration).
- `/scripts/` : Helper utilities and evaluation scripts.

---

python main.py --model_name resnet_tr --type predict --pretrained --test_model_name 54-81-0.5748.pth --test_file A0064

# Predict without pretrained weights
python main.py --model_name resnet_tr --type predict --test_model_name 54-81-0.5748.pth --test_file A0064

# Train a model from scratch (with optional pretrained base)
python main.py --model_name resnet_tr --max_epoch 100 --num_workers 10 --type train --pretrained
```

---

# Project Structure

- `/papers/` : Literature reviews and related research documentation.
- `/models/` : Saved model checkpoints.
- `/src/` : Core implementation (prediction, training, fuzzification, chatbot integration).
- `/scripts/` : Helper utilities and evaluation scripts.

---

# Credits

Developed and maintained by  
[@StaticOwl](https://www.github.com/StaticOwl) (Koustav Mallick)  

---

# Current Status

✅ Published at IEEE SoutheastCon 2025  
🚀 Future plans: Improve chatbot grounding, extend adaptive personalization, and integrate with clinical EHR systems.

---

# Citation

If you find this repository useful in your research, please cite:

```bibtex
@INPROCEEDINGS{mallick2025cardigpt,
  author={Mallick, Koustav and Singh, Neel and Hajiarbabi, Mohammedreza},
  title={{Cardi-GPT: An Expert ECG-Record Processing Chatbot}},
  booktitle={2025 IEEE SoutheastCon (SoutheastCon)},
  year={2025},
  pages={352-357},
  doi={10.1109/SoutheastCon56624.2025.10971509}
}
```
