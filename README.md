# UEA
The source code for the DASFAA 2021 paper: ***[Towards Entity Alignment in the Open World: An Unsupervised Approach](https://link.springer.com/chapter/10.1007/978-3-030-73194-6_19)***.

## Dependencies

* Python=3.6
* Tensorflow-gpu=1.13.1
* Scipy
* Numpy
* Scikit-learn
* python-Levenshtein

## Datasets
The original datasets are obtained from [DBP15K dataset](https://github.com/nju-websoft/BootEA),  [GCN-Align](https://github.com/1049451037/GCN-Align) and [JAPE](https://github.com/nju-websoft/JAPE).

Take the dataset DBP15K (ZH-EN) as an example, the folder "zh_en" contains:
* ent_ids_1: ids for entities in source KG (ZH);
* ent_ids_1_trans_goo: entities in source KG (ZH) with translated names;
* ent_ids_2: ids for entities in target KG (EN);
* ref_ent_ids: entity links for testing/validation;
* sup_ent_ids: entity links for training;
* triples_1: relation triples encoded by ids in source KG (ZH);
* triples_2: relation triples encoded by ids in target KG (EN);
* zh_vectorList.json: the input entity feature matrix initialized by word vectors;

### Semantic Information
Regarding the Semantic Information, we obtain the entity name embeddings for DBP15K from [RDGCN](https://github.com/StephanieWyt/RDGCN). You may also obtain from [here](https://share.weiyun.com/5qxLmEI).
Note that before running you need to place the `_vectorList.json` file under the corresponding directory.


## Running
* First generate the string similarity by running `python stringsim.py --lan "fr_en"` . The dataset could be chosen from `zh_en, ja_en, fr_en`
* Then run

```
python main.py --lan "fr_en"
```
* You may also directly run
```
bash auto.sh
```
> Due to the instability of embedding-based methods, it is acceptable that the results fluctuate a little bit  when running code repeatedly.

> If you have any questions about reproduction, please feel free to email to zengweixin13@nudt.edu.cn.

## Citation

If you use this model or code, please cite it as follows:
```
@inproceedings{DBLP:conf/dasfaa/ZengZTLLZ21,
  author    = {Weixin Zeng and
               Xiang Zhao and
               Jiuyang Tang and
               Xinyi Li and
               Minnan Luo and
               Qinghua Zheng}
  title     = {Towards Entity Alignment in the Open World: An Unsupervised Approach},
  booktitle = {DASFAA},
  pages     = {272--289},
  publisher = {Springer},
  year      = {2021},
}
```