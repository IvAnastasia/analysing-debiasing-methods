# Repository for Bachelor's thesis **Analyzing Methods for Debiasing of Russian Language Models**

## Abstract
Large language models (LLMs) are trained on vast amounts of unfiltered data which contain many instances of prejudice (in gender, nationality etc.) and inherit social biases present in the corpora. They have been shown to use these biases when applied to downstream tasks, reinforcing harmful social constructs (Sheng et al., 2019; Zhao et al., 2018).
Various debiasing techniques have been proposed to mitigate bias in LLMs and ensure their fairness in real-world applications. While some works are aimed to evaluate these techniques (Meade et al., 2022; Xie & Lukasiewicz, 2023), the implementations and comparisons of debiasing methods are rarely conducted on non-English languages. Therefore, it is crucial to evaluate bias-mitigating techniques across different languages to demonstrate the their stability and effectiveness.
In this study, we evaluate popular debiasing methods, including parameter-efficient techniques (adapter tuning, prefix-tuning, prompt tuning) and post-hoc approaches (SelfDebias, SentenceDebias), on several Russian medium-scale and large-scale language models. We use Russian Bias Detection Dataset RuBia (Grigoreva et al., 2024) to evaluate their performance after debiasing.

The contributions of this study are as follows: (i) a comprehensive evaluation of various debiasing techniques on Russian LLMs, (ii) analysis of the results, highlighting the most effective debiasing methods and providing explanations for observations; (iii) examination of the alignment of our findings with those for English LLMs; (iv) publication of code and experimental results on Github.

## Results
**Results of debiasing experiments conducted in gender/nationality/religion domain, model: BERT.**

| Domain      | SubDomain                  | BERT | BERT + Prefix Tune | BERT + Prompt Tune | BERT + Adapter Tune |
|-------------|----------------------------|:----:|:------------------:|:------------------:|:-------------------:|
| class       | all                        | 0.46 |        0.46        |        0.46        |         0.46        |
| class       | freeform_full              | 0.20 |        0.20        |        0.20        |         0.20        |
| class       | freeform_prof              | 0.70 |        0.70        |        0.70        |         0.70        |
| class       | template_wealth            | 0.43 |        0.43        |        0.43        |         0.43        |
| gender      | all                        | 0.48 |        0.48        |        0.48        |         0.48        |
| gender      | freeform_family_full       | 0.39 |        0.39        |        0.39        |         0.39        |
| gender      | freeform_family_stereotype | 0.37 |        0.37        |        0.37        |         0.37        |
| gender      | freeform_full              | 0.60 |        0.60        |        0.60        |         0.60        |
| gender      | freeform_generic           | 0.46 |        0.46        |        0.46        |         0.46        |
| gender      | freeform_job               | 0.64 |        0.64        |        0.64        |         0.64        |
| gender      | freeform_prof_full         | 0.35 |        0.35        |        0.35        |         0.35        |
| gender      | freeform_prof_stereotype   | 0.39 |        0.39        |        0.39        |         0.39        |
| gender      | template_assoc             | 0.61 |        0.61        |        0.61        |         0.61        |
| gender      | template_hetpos            | 0.52 |        0.52        |        0.52        |         0.52        |
| nationality | all                        | 0.65 |        0.65        |        0.65        |         0.65        |
| nationality | freeform_full              | 0.67 |        0.67        |        0.67        |         0.67        |
| nationality | freeform_immigrant         | 0.60 |        0.60        |        0.60        |         0.60        |
| nationality | template_assoc             | 0.66 |        0.66        |        0.66        |         0.66        |
| religion    | all                        | 0.77 |        0.77        |        0.77        |         0.77        |
| religion    | freeform_antisem           | 0.77 |        0.77        |        0.77        |         0.77        |

**Results of debiasing experiments conducted in gender/nationality/religion domain, model: GPT-2.**

| Domain      | SubDomain                  | GPT-2 | GPT-2 + Prefix Tune | GPT-2+ Prompt Tune | GPT-2+ Adapter Tune |
|-------------|----------------------------|:-----:|:-------------------:|:------------------:|:-------------------:|
| class       | all                        | 0.51  | 0.51                | 0.51               | 0.51                |
| class       | freeform_full              | 0.59  | 0.59                | 0.59               | 0.59                |
| class       | freeform_prof              | 0.48  | 0.48                | 0.48               | 0.48                |
| class       | template_wealth            | 0.47  | 0.47                | 0.47               | 0.47                |
| gender      | all                        | 0.57  | 0.57                | 0.57               | 0.57                |
| gender      | freeform_family_full       | 0.74  | 0.74                | 0.74               | 0.74                |
| gender      | freeform_family_stereotype | 0.74  | 0.74                | 0.74               | 0.74                |
| gender      | freeform_full              | 0.33  | 0.33                | 0.33               | 0.33                |
| gender      | freeform_generic           | 0.62  | 0.62                | 0.62               | 0.62                |
| gender      | freeform_job               | 0.46  | 0.46                | 0.46               | 0.46                |
| gender      | freeform_prof_full         | 0.65  | 0.65                | 0.65               | 0.65                |
| gender      | freeform_prof_stereotype   | 0.68  | 0.68                | 0.68               | 0.68                |
| gender      | template_assoc             | 0.35  | 0.35                | 0.35               | 0.35                |
| gender      | template_hetpos            | 0.43  | 0.43                | 0.43               | 0.43                |
| nationality | all                        | 0.54  | 0.54                | 0.54               | 0.54                |
| nationality | freeform_full              | 0.57  | 0.57                | 0.57               | 0.57                |
| nationality | freeform_immigrant         | 0.60  | 0.60                | 0.60               | 0.60                |
| nationality | template_assoc             | 0.47  | 0.47                | 0.47               | 0.47                |
| religion    | all                        | 0.41  | 0.41                | 0.41               | 0.41                |
| religion    | freeform_antisem           | 0.41  | 0.41                | 0.41               | 0.41                |

**Results of debiasing experiments conducted in gender, nationality and religion domain using SelfDebias (Schick et al., 2021) in comparison with parameter-efficient methods, model: BERT. Underlined scores are scores of domains which were the target of debiasing. Bold scores indicate the best result for every subdomain (in case there is a result between 0.35 and 0.65.**

| Domain      | SubDomain                  | BERT + parameter-efficient methods, all domains | BERT + SelfDebias, gender | BERT + SelfDebias, nationality | BERT + SelfDebias, religion |
|-------------|----------------------------|-------------------------------------------------|---------------------------|--------------------------------|-----------------------------|
| class       | all                        |                       0.46                      |            0.54           |              0.56              |             0.55            |
| class       | freeform_full              |                       0.20                      |            0.82           |              0.86              |             0.82            |
| class       | freeform_prof              |                       0.70                      |            0.29           |              0.29              |             0.31            |
| class       | template_wealth            |                       0.43                      |            0.58           |              0.61              |             0.58            |
| gender      | all                        |                       0.48                      |            0.55           |              0.57              |             0.53            |
| gender      | freeform_family_full       |                       0.39                      |            0.66           |              0.72              |             0.64            |
| gender      | freeform_family_stereotype |                       0.37                      |            0.67           |              0.70              |             0.63            |
| gender      | freeform_full              |                       0.60                      |            0.50           |              0.58              |             0.47            |
| gender      | freeform_generic           |                       0.46                      |            0.60           |              0.60              |             0.59            |
| gender      | freeform_job               |                       0.64                      |            0.32           |              0.28              |             0.32            |
| gender      | freeform_prof_full         |                       0.35                      |            0.59           |              0.53              |             0.59            |
| gender      | freeform_prof_stereotype   |                       0.39                      |            0.54           |              0.48              |             0.53            |
| gender      | template_assoc             |                       0.61                      |            0.59           |              0.74              |             0.50            |
| gender      | template_hetpos            |                       0.52                      |            0.39           |              0.43              |             0.39            |
| nationality | all                        |                       0.65                      |            0.42           |              0.44              |             0.39            |
| nationality | freeform_full              |                       0.67                      |            0.37           |              0.42              |             0.35            |
| nationality | freeform_immigrant         |                       0.60                      |            0.44           |              0.42              |             0.47            |
| nationality | template_assoc             |                       0.66                      |            0.47           |              0.48              |             0.38            |
| religion    | all                        |                       0.77                      |            0.16           |              0.19              |             0.07            |
| religion    | freeform_antisem           |                       0.77                      |            0.16           |              0.19              |             0.07            |

**Results of debiasing experiments conducted in gender, nationality and religion domain using SelfDebias (Schick et al., 2021) in comparison with parameter-efficient methods, model: GPT-2. Underlined scores are scores of domains which were the target of debiasing. Bold scores indicate the best result for every subdomain (in case there is a result between 0.35 and 0.65.**
| Domain      | SubDomain                  | GPT + parameter-efficient methods, all domains | GPT + SelfDebias, gender | GPT + SelfDebias, nationality | GPT + SelfDebias, religion |
|-------------|----------------------------|------------------------------------------------|--------------------------|-------------------------------|----------------------------|
| class       | all                        |                      0.51                      |           0.56           |              0.55             |            0.58            |
| class       | freeform_full              |                      0.59                      |           0.71           |              0.72             |            0.75            |
| class       | freeform_prof              |                      0.48                      |           0.44           |              0.41             |            0.44            |
| class       | template_wealth            |                      0.47                      |           0.56           |              0.57             |            0.58            |
| gender      | all                        |                      0.57                      |           0.47           |              0.50             |            0.47            |
| gender      | freeform_family_full       |                      0.74                      |           0.78           |              0.77             |            0.77            |
| gender      | freeform_family_stereotype |                      0.74                      |           0.74           |              0.74             |            0.74            |
| gender      | freeform_full              |                      0.33                      |           0.34           |              0.34             |            0.34            |
| gender      | freeform_generic           |                      0.62                      |           0.64           |              0.63             |            0.65            |
| gender      | freeform_job               |                      0.46                      |           0.17           |              0.25             |            0.20            |
| gender      | freeform_prof_full         |                      0.65                      |           0.20           |              0.38             |            0.26            |
| gender      | freeform_prof_stereotype   |                      0.68                      |           0.24           |              0.44             |            0.33            |
| gender      | template_assoc             |                      0.35                      |           0.30           |              0.22             |            0.19            |
| gender      | template_hetpos            |                      0.43                      |           0.52           |              0.52             |            0.52            |
| nationality | all                        |                      0.54                      |           0.38           |              0.37             |            0.32            |
| nationality | freeform_full              |                      0.57                      |           0.45           |              0.43             |            0.36            |
| nationality | freeform_immigrant         |                      0.60                      |           0.35           |              0.35             |            0.39            |
| nationality | template_assoc             |                      0.47                      |           0.33           |              0.33             |            0.22            |
| religion    | all                        |                      0.41                      |           0.78           |              0.80             |            0.09            |
| religion    | freeform_antisem           |                      0.41                      |           0.78           |              0.80             |            0.09            |

**Results of debiasing experiments conducted in gender, nationality and religion domain using SentDebias (Liang et al., 2020) in comparison with parameter-efficient methods, models: BERT and GPT-2. Underlined scores are scores of domains which were the target of debiasing. Bold scores indicate the best result for every subdomain (in case there is a result between 0.35 and 0.65.**

| Domain      | SubDomain                  | BERT + parameter-efficient methods, all domains | BERT + SentDeb, all domains | GPT-2 + parameter-efficient methods, all domains | GPT-2 + SentDeb, all domains |
|-------------|----------------------------|-------------------------------------------------|:---------------------------:|:------------------------------------------------:|:----------------------------:|
| class       | all                        |                       0.46                      |             0.46            | 0.51                                             |             0.46             |
| class       | freeform_full              |                       0.20                      |             0.20            | 0.59                                             |             0.20             |
| class       | freeform_prof              |                       0.70                      |             0.70            | 0.48                                             |             0.70             |
| class       | template_wealth            |                       0.43                      |             0.43            | 0.47                                             |             0.43             |
| gender      | all                        |                       0.48                      |             0.48            | 0.57                                             |             0.48             |
| gender      | freeform_family_full       |                       0.39                      |             0.39            | 0.74                                             |             0.39             |
| gender      | freeform_family_stereotype |                       0.37                      |             0.37            | 0.74                                             |             0.37             |
| gender      | freeform_full              |                       0.60                      |             0.60            | 0.33                                             |             0.60             |
| gender      | freeform_generic           |                       0.46                      |             0.46            | 0.62                                             |             0.46             |
| gender      | freeform_job               |                       0.64                      |             0.64            | 0.46                                             |             0.64             |
| gender      | freeform_prof_full         |                       0.35                      |             0.35            | 0.65                                             |             0.35             |
| gender      | freeform_prof_stereotype   |                       0.39                      |             0.39            | 0.68                                             |             0.39             |
| gender      | template_assoc             |                       0.61                      |             0.61            | 0.35                                             |             0.61             |
| gender      | template_hetpos            |                       0.52                      |             0.52            | 0.43                                             |             0.52             |
| nationality | all                        |                       0.65                      |             0.65            | 0.54                                             |             0.65             |
| nationality | freeform_full              |                       0.67                      |             0.67            | 0.57                                             |             0.67             |
| nationality | freeform_immigrant         |                       0.60                      |             0.60            | 0.60                                             |             0.60             |
| nationality | template_assoc             |                       0.66                      |             0.66            | 0.47                                             |             0.66             |
| religion    | all                        |                       0.77                      |             0.77            | 0.41                                             |             0.77             |
| religion    | freeform_antisem           |                       0.77                      |             0.77            | 0.41                                             |             0.77             |
