# Autor: Filipe de Paula Oliveira
# 
# Projeto 02 da Formação Cientista de Dados no curso Big Data Analytics com R
# e Microsoft Azure Machine Learning

# -- Importação de pacotes -- 

# Leitura e Manipulação de Dados
library(readr)
library(dplyr)
library(tidyr)
library(stringr)
library(reshape2)

# Análise Gráfica
library(ggplot2)
library(corrplot)
library(GGally)
library(lares)

# Machine Learning
library(caret)
library(rpart)
library(randomForest)
library(ranger)
library(xgboost)
library(strip)

# -- Análise do dataset -- 

# Foram fornecidos 3 datasets com informações adicionais:

# cliente_tabla.csv -> nomes dos clientes com join em Cliente_ID
# producto_tabla.csv -> nome dos produtos com join em Proucto_ID
# town_state.csv — cidade e estado com join em Agencia_ID

# Ao meu ver não há necessidade dos dois primeiros, pois representam a mesma informação que o respectivo ID
# Já cidade e estado podem fornecer nos informações adicionais, então serão utilizados

# -- Leitura do dataset --

colTypes <- cols(Semana = 'i', Agencia_ID = 'i', Canal_ID = 'i', Ruta_SAK = 'i',
                 Cliente_ID = 'i', Producto_ID = 'i', Venta_uni_hoy = 'i',
                 Venta_hoy = 'd', Dev_uni_proxima = 'i', Dev_proxima = 'd',
                 Demanda_uni_equil = 'i', Town = 'c', State = 'c')

# Lendo train dataset real, join town_cities e obtendo 1.000.000 linhas aleatórias

# df_orig <- read_csv("grupo-bimbo-inventory-demand/train.csv",
#                    col_types = colTypes)
# df_cities <- read_csv("grupo-bimbo-inventory-demand/town_state.csv")
# df_subset <- df_orig[sample(nrow(df_orig), size = 1000000), ]
# df_merged <- merge(df_subset, df_cities, by = 'Agencia_ID')

# Gravando num arquivo para não ter que executar toda vez

# write_csv(df_merged, "trainTransf.csv")
# remove(df_orig)
# remove(df_cities)
# remove(df_subset)
# remove(df_merged)

# Train dataset transformado final
df_inicial <- read_csv("trainTransf.csv", col_types = colTypes)

# -- Análise exploratória -- (pode ser comentada pra rodar apenas o modelo)

# Descrição do dataset

head(df_inicial)
str(df_inicial)
summary(df_inicial)

# Gráfico da contagem de valores unique

count_uniques <- df_inicial %>%
  summarise_all(n_distinct) %>%
  dplyr::select(-Venta_hoy, -Demanda_uni_equil) %>%
  melt()
ggplot(count_uniques) + 
  geom_bar(stat="identity") +
  aes(variable, value, fill = variable) +
  theme_bw() +
  geom_text(aes(variable, value + 15000, label = value, fill = NULL)) +
  labs(title = "Quantidade de Valores Únicos por Variável",
       x = "",
       y = "Count Unique")

# Gráfico de correlações

only_numeric <- select(df_inicial, -Town, -State)
corrplot(cor(only_numeric),
         method = 'color',
         title = 'Gráfico de Correlações',
         mar=c(0,0,1,0))

# Histogramas

ggplot(gather(only_numeric), aes(x=value)) +
  geom_histogram(bins=15, fill = 'turquoise4') +
  facet_wrap(~key, scales = 'free_x')

# Plot das maiores ocorrências da variável target

target_unique <- count(df_inicial, vars = Demanda_uni_equil)
target_unique <- target_unique[order(target_unique$n, decreasing = T),]

ggplot(target_unique[1:20,]) + 
  geom_bar(stat="identity", fill = 'forestgreen') +
  aes(x = reorder(vars, -n), y = n) +
  theme_bw() +
  labs(title = "Maiores ocorrências da variável target",
       x = "Target",
       y = "Ocorrências")

# Scatterplots com variável target (primeiros 1000)

only_numeric[1:1000,] %>%
  gather(-Demanda_uni_equil, key = "var", value = "value") %>% 
  ggplot(aes(x = value, y = Demanda_uni_equil, alpha = 0.05)) +
  geom_point() +
  facet_wrap(~ var, scales = "free") +
  theme_bw() +
  ggtitle("Scatterplot de var x variável target (apenas primeiros 1000)")

# -- Pré-Processamento --

# Verificando valores NA

table(is.na(df_inicial))

# Essa transformação abaixo na verdade piorou o modelo, por isso foi deixado comentada
# O modelo foi revertido para utilizar a coluna original de Town

# Arrumando Town para tentar representar cidade, e não a agência em si

# Verificando os valores existentes
# table(df_inicial$Town)

# Função para dividir a String por espaço e retornar apenas os últimos 2 valores

# splitter <- function(x) {
#   splitted = unlist(strsplit(x, split = ' '))
#   if (length(splitted) == 6) {
#     return( sprintf("%s %s",splitted[5],splitted[6]) )
#   }
#   if (length(splitted) == 5) {
#     return( sprintf("%s %s",splitted[4],splitted[5]) )
#   }
#   if (length(splitted) == 4) {
#     return( sprintf("%s %s",splitted[3],splitted[4]) )
#   }
#   else if(length(splitted) == 3) {
#     return( sprintf("%s %s",splitted[2],splitted[3]) )
#   }
#   else if(length(splitted) == 2) {
#     return( sprintf("%s",splitted[2]) )
#   }
#   else
#     return( sprintf("%s",splitted[1]) )
# }

# Removendo alguns caracteres encontrados e aplicando a função de split

# df_pre1 <- str_replace_all(df_inicial$Town, 'AG. ', '')
# df_pre1 <- str_replace_all(df_pre1, 'BIMBO', '')
# df_pre1 <- str_replace_all(df_pre1, ' INSTITUCIONALES', '')
# df_pre1 <- str_replace_all(df_pre1, ' II', '')
# df_pre1 <- str_replace_all(df_pre1, ' I', '')
# df_pre1 <- str_replace_all(df_pre1, '2', '')
# df_pre1 <- str_replace_all(df_pre1, '1', '')
# df_pre1 <- sapply(df_pre1, splitter)
# 
# table(df_pre1)

# Normalização das variáveis numéricas e adicionando Town e State

normalizar <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

df_norm <- as.data.frame(lapply(only_numeric, normalizar)) %>%
  mutate(Town = df_inicial$Town, State = df_inicial$State)

# Sampling 

splitIndex <- createDataPartition(y = df_norm$Demanda_uni_equil, p = 0.7, list = FALSE)
df_train <- df_norm[splitIndex,]
df_test <- df_norm[-splitIndex,]

# -- Machine Learning --

# Modelo #1 (Regressão Linear)

modelo_lr <- lm(Demanda_uni_equil ~ ., df_train, y = FALSE, model = FALSE)
summary(modelo_lr)

predicoes_lr <- predict(modelo_lr, df_test)
rmse_lr <- RMSE(predicoes_lr, df_test$Demanda_uni_equil)
r2_lr <- R2(predicoes_lr, df_test$Demanda_uni_equil)

# Modelo #2 (CART)

modelo_cart <- rpart(Demanda_uni_equil ~ ., df_train, method = 'anova')
modelo_cart

predicoes_cart <- predict(modelo_cart, df_test)
rmse_cart <- RMSE(predicoes_cart, df_test$Demanda_uni_equil)
r2_cart <- R2(predicoes_cart, df_test$Demanda_uni_equil)

# Modelo #3 (Random Forest)

modelo_rf <- ranger(Demanda_uni_equil ~ ., df_train)
modelo_rf

predicoes_rf <- predict(modelo_rf, df_test)$predictions
rmse_rf <- RMSE(predicoes_rf, df_test$Demanda_uni_equil)
r2_rf <- R2(predicoes_rf, df_test$Demanda_uni_equil)

# Modelo #4 (XGBoost)

xgb_train <- xgb.DMatrix(data = data.matrix(select(df_train, -Demanda_uni_equil)),
                         label = df_train$Demanda_uni_equil)
xgb_test <- xgb.DMatrix(data = data.matrix(select(df_test, -Demanda_uni_equil)),
                        label = df_test$Demanda_uni_equil)

modelo_xgb <- xgboost(data=xgb_train, max.depth=3, nrounds=100)
modelo_xgb

predicoes_xgb <- predict(modelo_xgb, xgb_test)
rmse_xgb <- RMSE(predicoes_xgb, df_test$Demanda_uni_equil)
r2_xgb <- R2(predicoes_xgb, df_test$Demanda_uni_equil)

# Plot dos Modelos no disco (um pouco lento, pode comentar -> já estão em /plots/)

lares::mplot_lineal(tag = df_test$Demanda_uni_equil,
                    score = predicoes_lr,
                    subtitle = "Linear Regression",
                    save = T,
                    subdir = 'plots',
                    file_name = 'modelLR')

lares::mplot_lineal(tag = df_test$Demanda_uni_equil,
                    score = predicoes_cart,
                    subtitle = "CART",
                    save = T,
                    subdir = 'plots',
                    file_name = 'modelCART')

lares::mplot_lineal(tag = df_test$Demanda_uni_equil,
                    score = predicoes_rf,
                    subtitle = "Random Forest",
                    save = T,
                    subdir = 'plots',
                    file_name = 'modelRF')

lares::mplot_lineal(tag = df_test$Demanda_uni_equil,
                    score = predicoes_xgb,
                    subtitle = "XGBoost",
                    save = T,
                    subdir = 'plots',
                    file_name = 'modelXGB')

# Comparação dos Modelos

resumo <- data.frame(Modelo = c('Linear Regression','CART','Random Forest','XGBoost'),
                     RMSE = c(rmse_lr,rmse_cart,rmse_rf,rmse_xgb),
                     R2 = c(r2_lr, r2_cart, r2_rf, r2_xgb))

ggplot(resumo) +
  geom_bar(stat = 'identity', fill = 'forestgreen') +
  aes(x = Modelo, y = R2) +
  geom_text(aes(Modelo, R2 + 0.05, label = round(R2,4), fill = NULL)) +
  theme_bw() +
  ggtitle("Comparação do R² de cada modelo")

ggplot(resumo) +
  geom_bar(stat = 'identity', fill = 'forestgreen') +
  aes(x = Modelo, y = RMSE) +
  geom_text(aes(Modelo, RMSE + 0.0001, label = round(RMSE,6), fill = NULL)) +
  theme_bw() +
  ggtitle("Comparação do RMSE de cada modelo")

# Limpando partes desnecessárias do modelo pra poupar espaço em disco no .Rdata

strip(modelo_lr, keep = "predict")