# Детектор нейтрального лица

Содержание:
* [Общий принцип работы](#общий-принцип-работы)
* [Детектирование лиц](#детектирование-лиц)
* [Классификация областей интереса](#классификация)
* [Формирование выборки](#формирование-выборки)
* [Метрики качества](#метрики-качества)
* [Время работы](#время-работы)
* [Ипользование модели](#использование-модели)

## Общий принцип работы

1. На фотографии детектируются лица с помощью библиотеки dlib.
2. На найденных лица осуществляется локализация ключевых точек с помощью модели из библиотеки dlib.
3. Отдельно обученная нейронная сеть классифицирует область в районе лица на два класса: "улыбка", "открытый рот".

Инструмент реализован с помощью языка python3.6, с использованием следующих библиотек: dlib, OpenCV, MXNet. Все данные библиотеки реализованы на языке C++, что позволяет перенести полученное решение на мобильные устройства, без изменения модели.

## Детектирование лиц

Для детектирования лиц на изображениях использован детектор из библиотеки [dlib](https://github.com/davisking/dlib).

Для оценки качества работы детектора лиц, была осуществлена проверка работы детектора лиц на 300 демонстрационных фотографиях:

| Model | TP | FP | FN | Precision | Recall |
| --- | --- | --- | --- | --- | --- |
| dlib.frontal_face_detector | 298 | 1 | 2 | 0.996655518 | 0.993333333 |

Данная модель имеет далеко не самое идеальное качество, но позволяет покрыть большинство тестовых примеров с приемлемой скоростью работы, что позволяет интегрировать решения подобного рода на устройства с ограниченными вычислительными ресурсами.

## Формирование выборки

По демонстрационным примерам, характеризующим целевое распределение, видно, что основная часть данных это фронтальные фотографии лица, селфи.

Существует большое количество размеченных датасетов с лицами, имеющие те или иные недостатки:
* Отсутствие необходимой разметки (наличие "улыбки" является стандартом для [facial expression](https://www.behance.net/gallery/10675283/Facial-Expression-Public-Databases) датасетов, а "открытый рот" встречается значительно реже)
* Отличие распределения от целевого (например, фотографии в constrained environment у старых датасетов)
* Лицензионные ограничение (non-commercial usage)

Из общедоступных датасетов, наиболее схожее распределение имеет:  
[http://crcv.ucf.edu/data/Selfie/](http://crcv.ucf.edu/data/Selfie/). Также дополнительным плюсом является наличие в данном датасете размеченных атрибутов "открытый рот", "улыбка".

Большинство вышеперечисленных ограничений легко преодолевается с помощью собственной разметки (остается за рамками данной работы).  В дальнейшем, при построении моделей будем использовать данные из `Selfie` датасета.

## Метрики качества

Для измерения качества построенной модели, использовалась метрика ROC-AUC.
Результаты на демонстрационных примерах, при обучении на стороннем датасете:

| Expression | ROC-AUC
| --- | --- |
| Smile | 0.96
| Mouth open | 0.95 

## Время работы

| Stage | Время работы на "i7-4500U CPU @ 1.80GHz", ms |
| --- | --- |
| Face Detection | 28 |
| Landmark detection | 2.5 |
| Facial Expression Classification | 10 |

## Использование модели

Язык: python 3.6

Для автоматизации установки необходимых зависимостей используется Docker.

Порядок действий для запуска модели детектирования:

1. Установить Docker. https://docs.docker.com/install/

1. Собрать образ:

```
docker build -t face_neutrality -f Dockerfile .
```

В процессе сборки образа, будут установлены сторонние зависимости, а также загружены необходимые модели, и сторонний датасет.

2. Зайти в собранный образ, с монтированием директории хостовой системы, содержащей изображения:

```
docker run -v /home/u1234x1234/example_data/:/example_data/ -ti face_neutrality bash
```

3. Запустить скрипт классификации:

```
python classify_expression.py --in_dir=example_data/images/ --out_dir=filtered_images
```

Требуемая функция, производящая разбиение списка изображения на два класса, реализована в файле `classify_expression.py`.

## Обучение

Сторонние данные, используемы для обучения, автоматически загружаются, во время сборки Docker образа. Поэтому отдельно можно ничего не скачивать.

Для обучения модели классификации (улыбка, открытый рот), необходимо:

1. Подготовить данные для обучения:
```
python crop_mouth.py --in_path=Selfie-dataset/images/ --out_path=mouth_selfie/
python crop_mouth.py --in_path=example_data/images/ --out_path=mouth_test/
```

2. Запустить скрипт обучения

```
train_face_classifier.py
```

Результирующая модель будет сохранена в файлы с префиксом `expression_classifier`.
