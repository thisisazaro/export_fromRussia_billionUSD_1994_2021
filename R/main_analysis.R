
library(readxl)
library(forecast)
library(ggplot2)
library(tseries)
library(tidyverse)
library(tsibble)
library(fable)
library(randomForest)
library(strucchange)

data <- read_excel("data/export_fromRussia_billionUSD.xlsx")
str(data)
colnames(data) <- c("Year", "Month", "Export")

# Преобразуем в временной ряд
data_ts <- ts(data$Export, start = c(min(data$Year), 1), frequency = 12)

# Очистка выбросов и замена пропущенных значений
data_ts <- tsclean(data_ts)

autoplot(data_ts) + ggtitle("Общий объем экспорта из РФ (млрд USD)")

# Обычные и сезонные разности
diff_ts <- diff(data_ts)  # Обычная разность
diff_seasonal_ts <- diff(data_ts, lag = 12)  # Сезонная разность

autoplot(diff_ts) + ggtitle("Временной ряд после обычного дифференцирования")
autoplot(diff_seasonal_ts) + ggtitle("Временной ряд после сезонного дифференцирования")

# Компоненты ряда
decomp <- stl(data_ts, s.window = "periodic")
autoplot(decomp)

# ACF и PACF
acf(data_ts, main = "ACF исходного ряда")
pacf(data_ts, main = "PACF исходного ряда")

acf(diff_ts, main = "ACF обычной разности")
pacf(diff_ts, main = "PACF обычной разности")

acf(diff_seasonal_ts, main = "ACF сезонной разности")
pacf(diff_seasonal_ts, main = "PACF сезонной разности")


# Проверка стационарности
adf.test(data_ts)
adf.test(diff_ts)
adf.test(diff_seasonal_ts)

# Преобразование при необходимости
log_ts <- log(data_ts)  # Логарифмируем ряд для стабилизации дисперсии
adf.test(diff(log_ts))  # Повторная проверка

# Разделение на обучающую и тестовую выборку
train_ts <- window(data_ts, end = c(2018, 12))
test_ts <- window(data_ts, start = c(2019, 1))

# Оценка моделей
naive_model <- naive(train_ts, h = length(test_ts))
ets_model <- ets(train_ts)
sarima_model <- auto.arima(train_ts, seasonal = TRUE)
theta_model <- thetaf(train_ts, h = length(test_ts))
train_df <- data.frame(Time = as.numeric(time(train_ts)), Export = as.numeric(train_ts))
train_df <- na.omit(train_df)
rf_model <- randomForest(Export ~ Time, data = train_df)
avg_forecast <- (forecast(ets_model, h = length(test_ts))$mean + forecast(sarima_model, h = length(test_ts))$mean) / 2

# Строим прогнозы для тестовой выборки
forecast_naive <- forecast(naive_model)
forecast_ets <- forecast(ets_model, h = length(test_ts))
forecast_sarima <- forecast(sarima_model, h = length(test_ts))
forecast_theta <- forecast(theta_model)

# Оцениваем точность прогнозов
accuracy_naive <- tryCatch(accuracy(forecast(naive_model, h = length(test_ts)), test_ts), error = function(e) rep(NA, 6))
accuracy_ets <- tryCatch(accuracy(forecast(ets_model, h = length(test_ts)), test_ts), error = function(e) rep(NA, 6))
accuracy_sarima <- tryCatch(accuracy(forecast(sarima_model, h = length(test_ts)), test_ts), error = function(e) rep(NA, 6))
accuracy_theta <- tryCatch(accuracy(forecast(theta_model, h = length(test_ts)), test_ts), error = function(e) rep(NA, 6))
print("Accuracy Naive:")
print(accuracy_naive)
print("Accuracy ETS:")
print(accuracy_ets)
print("Accuracy SARIMA:")
print(accuracy_sarima)
print("Accuracy Theta:")
print(accuracy_theta)

model_scores <- data.frame(
        Model = c("Naive", "ETS", "SARIMA", "Theta"),
        RMSE = c(as.numeric(accuracy_naive["Test set", "RMSE"]),
                 as.numeric(accuracy_ets["Test set", "RMSE"]),
                 as.numeric(accuracy_sarima["Test set", "RMSE"]),
                 as.numeric(accuracy_theta["Test set", "RMSE"])),
        MAE = c(as.numeric(accuracy_naive["Test set", "MAE"]),
                as.numeric(accuracy_ets["Test set", "MAE"]),
                as.numeric(accuracy_sarima["Test set", "MAE"]),
                as.numeric(accuracy_theta["Test set", "MAE"]))
)

model_scores <- na.omit(model_scores)
model_scores <- model_scores[is.finite(model_scores$RMSE), ]
model_scores

if (nrow(model_scores) > 0 && all(is.finite(model_scores$RMSE))) {
        best_model_name <- model_scores$Model[which.min(model_scores$RMSE)]
        cat("Лучшая модель по RMSE:", best_model_name, "\n")
        
        best_model <- switch(best_model_name,
                             "Naive" = naive_model,
                             "ETS" = ets_model,
                             "SARIMA" = sarima_model,
                             "Theta" = theta_model
        )
        
        checkresiduals(best_model)
        forecast_best <- forecast(best_model, h = 24)
        autoplot(forecast_best) + ggtitle("Прогноз на 2 года вперед")
} else {
        cat("Ошибка: RMSE содержит NA или Inf. Проверьте данные.\n")
}


forecast_best

