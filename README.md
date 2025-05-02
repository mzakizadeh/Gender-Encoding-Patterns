# Gender Encoding Patterns

This repository contains the replication code for our publication in TrustNLP 2025 titled [Gender Encoding Patterns in Pretrained Language Model Representations](https://arxiv.org/pdf/2503.06734).

> **Abstract:** Gender bias in pretrained language models (PLMs) poses significant social and 
> ethical challenges. Despite growing awareness, there is a lack of comprehensive investigation into
> how different models internally represent and propagate such biases. This study adopts an 
> information-theoretic approach to analyze how gender biases are encoded within various 
> encoder-based architectures. We focus on three key aspects: identifying how models encode gender 
> information and biases, examining the impact of bias mitigation techniques and fine-tuning on the 
> encoded biases and their effectiveness, and exploring how model design differences influence the 
> encoding of biases. Through rigorous and systematic investigation, our findings reveal a 
> consistent pattern of gender encoding across diverse models. Surprisingly, debiasing techniques 
> often exhibit limited efficacy, sometimes inadvertently increasing the encoded bias in internal 
> representations while reducing bias in model output distributions. This highlights a disconnect 
> between mitigating bias in output distributions and addressing its internal representations. 
> This work provides valuable guidance for advancing bias mitigation strategies and fostering the 
> development of more equitable language models.

This code is an extension of the implementation of the works of [Orgad et al.](https://aclanthology.org/2022.naacl-main.188.pdf) and [Mendelson and Belinkov](https://arxiv.org/pdf/2109.04095).

## Contact Information

For any inquiries regarding the study please feel free to contact me:
- Name: Mahdi Zakizadeh
- Email: mzakizadeh.me@gmail.com

## Citation

```
@inproceedings{zakizadeh-pilehvar-2025-gender,
    title = "Gender Encoding Patterns in Pretrained Language Model Representations",
    author = "Zakizadeh, Mahdi  and
      Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 5th Workshop on Trustworthy NLP (TrustNLP 2025)",
    month = may,
    year = "2025",
    address = "Albuquerque, New Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.trustnlp-main.31/",
    pages = "489--500",
    ISBN = "979-8-89176-233-6",
}
```
