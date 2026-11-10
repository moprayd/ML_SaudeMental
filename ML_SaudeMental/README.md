# KNN para identificação de indicação de transtorno mental (AMI) no NSDUH 2021–2024

Código do trabalho **"Aplicação de técnicas de Machine Learning na análise de saúde mental no contexto da Segurança da Informação"** (Fatec Ourinhos, Curso Superior de Tecnologia em Segurança da Informação).

O script treina um classificador **K-Nearest Neighbors** (implementado em PyTorch, com GPU opcional) para prever o indicador binário **AMI** (*any mental illness* no último ano) em adultos, a partir do perfil sociodemográfico e do uso de álcool, maconha, cocaína, crack e heroína. As escalas clínicas usadas pela SAMHSA para derivar o AMI **não** entram como preditores, para evitar circularidade (*data leakage*).

> Este repositório é material de pesquisa. O modelo tem sinal preditivo moderado e **não** deve ser usado para triagem individual de pessoas, nem em contexto de emprego.

## Resultados (conjunto de teste, n = 37.365)

| Estratégia | Acurácia | Ac. balanceada | Precisão AMI | Recall AMI | F1 AMI | ROC-AUC | PR-AUC | Tempo (GPU) |
|---|---|---|---|---|---|---|---|---|
| Sem balanceamento (`none`) | 0,6229 | 0,6459 | 0,40 | 0,70 | 0,5089 | 0,7040 | 0,4746 | 6m31s |
| Sobreamostragem aleatória (`random`) | 0,5987 | 0,6410 | 0,39 | 0,74 | 0,5069 | 0,7013 | 0,4699 | 9m09s |
| SMOTE (`smote`) | 0,6139 | 0,6429 | 0,39 | 0,71 | 0,5068 | 0,6995 | 0,4665 | 9m02s |

- Configuração escolhida por validação cruzada no treino (5 folds): distância de Manhattan, k = 200, ponderação uniforme, limiar de decisão 0,255 (`none`), 0,445 (`random`) e 0,460 (`smote`).
- Referências: prevalência de AMI no teste = 0,2799 (PR-AUC de um classificador aleatório); classe majoritária = acurácia 0,7201 com recall nulo; "sempre AMI" = F1 0,4374.
- Matriz de confusão do `none` (TN, FP, FN, TP): 15.975 / 10.932 / 3.157 / 7.301 (`docs/matriz_confusao_none.jpeg`).
- O oversampling praticamente só desloca o limiar de decisão: no KNN, ajustar o limiar chega ao mesmo ponto de operação sem o custo extra de tempo.

Os números podem variar na terceira ou quarta casa decimal entre máquinas, por causa de `float32` em GPU e do desempate entre vizinhos à mesma distância.

## Estrutura

```
.
├── Ml_MHealth_final.py     # pipeline completo (dados -> CV -> teste -> métricas)
├── requirements.txt
├── docs/
│   └── matriz_confusao_none.jpeg
├── LICENSE
└── README.md
```

## Como reproduzir

0. **Clone o repositório**:
   ```bash
   git clone https://github.com/moprayd/ML_SaudeMental.git
   cd ML_SaudeMental
   ```
1. **Baixe o dado** (não está no repositório): arquivo de uso público combinado **NSDUH 2021–2024**, em formato tab-delimited, na página de *data files* da SAMHSA: <https://www.samhsa.gov/data/data-we-collect/nsduh-national-survey-drug-use-health/datafiles>. Extraia e deixe o arquivo como `NSDUH_2021_2024_Tab.tsv` na mesma pasta do script (232.441 linhas e 2.638 colunas).
2. **Instale as dependências** (de preferência em um ambiente virtual):
   ```bash
   pip install -r requirements.txt
   ```
   Para usar a GPU, instale o PyTorch com CUDA pelo seletor de <https://pytorch.org/get-started/locally/>.
3. **Rode** (o argumento escolhe o tipo de oversampling, sempre aplicado só ao treino):
   ```bash
   python Ml_MHealth_final.py          # baseline
   python Ml_MHealth_final.py random
   python Ml_MHealth_final.py smote
   ```

Saídas: resultado no terminal (métricas, relatório de classificação, matriz de confusão, ROC-AUC, PR-AUC, top 5 da busca), `matriz_confusao_<modo>.png` e `tempos_execucao.csv` (uma linha por execução, com tempo por etapa e média).

Conferência rápida de que o dado e o filtro estão corretos: **186.823** adultos válidos, **149.458** no treino, **37.365** no teste e prevalência de AMI de **0,2799**.

## Metodologia em resumo

- **Amostra:** adultos (`AGE3` de 4 a 11) com `AMIPY` e `SMIPY` válidos e consistentes.
- **Alvo:** `AMIPY` (0 = sem indicação, 1 = AMI).
- **Pré-processamento:** códigos especiais viram ausentes; imputação (mediana para numéricas e ordinais, moda para categóricas e binárias); one-hot nas categóricas; padronização (z-score). Ajustado só no treino de cada fold.
- **Divisão:** 80% treino / 20% teste, estratificada, `random_state = 42`.
- **Busca em grade (5 folds estratificados):** métrica (euclidiana, Manhattan), k ∈ {15, 30, 50, 70, 100, 150, 200} e ponderação (uniforme, por distância).
- **Limiar de decisão:** escolhido nas predições *out-of-fold* do treino para maximizar o F1 da classe AMI. O teste não é usado para escolher nada.
- **Oversampling:** aleatório ou SMOTE, aplicado apenas ao treino de cada fold e ao treino final. Validação e teste mantêm a prevalência original.

## Limitações

- AMI é uma estimativa estatística da SAMHSA, não um diagnóstico clínico.
- Pesos amostrais do levantamento não foram aplicados: o objetivo é a predição individual, não a estimação populacional.
- O desempenho reflete a exclusão deliberada das escalas clínicas.
- O KNN guarda o conjunto de treino para predizer; implantar o modelo equivale a distribuir os dados dos respondentes, o que é um ponto de atenção de privacidade.
- Confira nomes de variáveis e códigos especiais no codebook 2021–2024 antes de reutilizar o código com outro recorte.
- Os dados são dos EUA; não há validação externa em outras populações.

## Dados e privacidade

O NSDUH é distribuído pela SAMHSA como arquivo de uso público com controle de divulgação estatística. Siga os termos de uso da SAMHSA ao baixar e citar o dado. O arquivo `.tsv` está no `.gitignore` e não deve ser versionado.

## Autoria

Adriel Fiorilo da Silva Jardim · Orientador: Prof. Me. Eduardo Alves Moraes · Fatec Ourinhos.

Código sob licença MIT (veja `LICENSE`).
