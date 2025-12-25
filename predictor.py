import pandas as pd
import numpy as np
import os
import warnings
import joblib
import re
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

VLADIMIR_FILE = "data/input/Владимирская область_обработка.xlsx"
MODEL_PATH = "data/output/population_model_combined.joblib"
SCALER_PATH = "data/output/scaler_combined.joblib"
OUTPUT_DIR = "data/output"
VLADIMIR_RESULTS = os.path.join(OUTPUT_DIR, "vladimir_population_predictions.xlsx")

def load_and_prepare_vladimir_data():

    df = pd.read_excel(VLADIMIR_FILE)
    print(f"Владимирская область: {len(df)} строк, {len(df.columns)} колонок")
    print(f"Колонки: {list(df.columns)}")
    
    column_mapping = {
        '@id': 'house_id',
        '@lat': 'lat',
        '@lon': 'lon',
        'addr:region': 'parsed_region',
        'addr:city': 'parsed_city',
        'addr:street': 'parsed_street',
        'addr:housenumber': 'parsed_house',
        'building': 'building_type',
        'building:levels': 'building_levels',
        'building:flats': 'building_flats'
    }
    
    df = df.rename(columns=column_mapping)
    
    df['region'] = 'Владимирская область'
    
    df = parse_osm_address(df)
    
    return df

def parse_osm_address(df):
 
    df['house_number'] = df['parsed_house'].apply(extract_house_number)
    
    df['settlement_type'] = df['parsed_city'].apply(detect_settlement_type)
    
    if 'building_type' in df.columns:
        df['parsed_building_type'] = df['building_type'].apply(standardize_building_type)
    else:
        df['parsed_building_type'] = 'residential'
    
    print(f"Типы зданий:")
    if 'parsed_building_type' in df.columns:
        for btype, count in df['parsed_building_type'].value_counts().head(10).items():
            print(f"  {btype}: {count}")
    
    print(f"Уникальных городов: {df['parsed_city'].nunique()}")
    print(f"Уникальных улиц: {df['parsed_street'].nunique()}")
    
    return df

def extract_house_number(house_str):
    if pd.isna(house_str) or not isinstance(house_str, str):
        return 1
    
    numbers = re.findall(r'\d+', str(house_str))
    if numbers:
        return int(numbers[0])
    return 1

def detect_settlement_type(settlement):
    if pd.isna(settlement) or not isinstance(settlement, str):
        return 'city'
    
    settlement_lower = settlement.lower()
    
    if any(word in settlement_lower for word in ['город', 'г.', 'г ', 'город']):
        return 'city'
    elif any(word in settlement_lower for word in ['село', 'с.', 'с ', 'село']):
        return 'village'
    elif any(word in settlement_lower for word in ['деревня', 'дер.', 'д.', 'д ']):
        return 'rural'
    elif any(word in settlement_lower for word in ['поселок', 'пос.', 'пгт', 'п.']):
        return 'township'
    elif any(word in settlement_lower for word in ['район', 'округ', 'р-н']):
        return 'district'
    else:
        return 'city' 

def standardize_building_type(btype):
    if pd.isna(btype) or not isinstance(btype, str):
        return 'residential'
    
    btype_lower = btype.lower()
    
    if any(word in btype_lower for word in ['apartment', 'apartments', 'residential']):
        return 'apartments'
    elif any(word in btype_lower for word in ['house', 'detached']):
        return 'house'
    elif any(word in btype_lower for word in ['dormitory', 'dorm']):
        return 'dormitory'
    elif any(word in btype_lower for word in ['gatehouse', 'garage']):
        return 'gatehouse'
    elif any(word in btype_lower for word in ['commercial', 'office', 'shop']):
        return 'commercial'
    elif any(word in btype_lower for word in ['industrial', 'factory', 'warehouse']):
        return 'industrial'
    else:
        return 'residential'

def create_features_for_vladimir(df):

    result = df.copy()
    
    result['coord_sum'] = result['lon'] + result['lat']
    result['coord_diff'] = result['lon'] - result['lat']
    
    le_region = LabelEncoder()

    regions = ['Пермский край', 'Свердловская область', 'Владимирская область']
    le_region.fit(regions)
    result['region_encoded'] = le_region.transform(result['region'])
    
    le_settlement = LabelEncoder()
    settlement_types = ['city', 'village', 'rural', 'township', 'district', 'region', 'other']
    le_settlement.fit(settlement_types)
    result['settlement_type_encoded'] = le_settlement.transform(
        result['settlement_type'].fillna('city')
    )
    
    result['street_name_length'] = result['parsed_street'].apply(
        lambda x: len(str(x)) if pd.notna(x) else 0
    )
    
    result['house_number'] = result['house_number'].fillna(1).astype(int)
    result['house_is_even'] = (result['house_number'] % 2 == 0).astype(int)
    
    result['house_category'] = pd.cut(
        result['house_number'],
        bins=[0, 10, 50, 100, float('inf')],
        labels=['small', 'medium', 'large', 'very_large']
    )
    
    le_house_cat = LabelEncoder()
    house_categories = ['small', 'medium', 'large', 'very_large']
    le_house_cat.fit(house_categories)
    result['house_category_encoded'] = le_house_cat.transform(result['house_category'])
    
    # Для Владимирской области вычисляем плотность на основе собственных данных
    result['lat_rounded'] = (result['lat'] * 100).round() / 100
    result['lon_rounded'] = (result['lon'] * 100).round() / 100
    
    density = result.groupby(['lat_rounded', 'lon_rounded']).size()
    density_dict = density.to_dict()
    result['density'] = result.apply(
        lambda row: density_dict.get((row['lat_rounded'], row['lon_rounded']), 1),
        axis=1
    )
    
    # Оценочная площадь здания
    result['estimated_area'] = 50 + result['house_number'] * 2 + result['density'] * 10
    result['estimated_area'] = result['estimated_area'].clip(lower=30, upper=5000)
    
    result['likely_residential'] = result['density'].apply(lambda x: 1 if x > 5 else 0)
    
    if 'building_levels' in result.columns:
        result['building_levels'] = pd.to_numeric(
            result['building_levels'], errors='coerce'
        ).fillna(2)  # По умолчанию 2 этажа
    else:
        result['building_levels'] = 2
    
    # Площадь здания (оценочная)
    result['building_area'] = result['estimated_area'] * result['building_levels']
    
    result = result.drop(['lat_rounded', 'lon_rounded'], axis=1, errors='ignore')
    
    print(f"Создано {len(result.columns)} признаков")
    
    return result

def get_features_for_prediction(df):
   
    base_features = [
        'lon', 'lat',
        'coord_sum', 'coord_diff',
        'region_encoded',
        'settlement_type_encoded',
        'street_name_length',
        'house_number',
        'house_is_even',
        'house_category_encoded',
        'density',
        'estimated_area',
        'building_levels',
        'building_area',
        'likely_residential'
    ]
    
    missing_features = [f for f in base_features if f not in df.columns]
    if missing_features:
        print(f"ВНИМАНИЕ: Отсутствуют признаки: {missing_features}")
        for feature in missing_features:
            df[feature] = 0
    
    available_features = [col for col in base_features if col in df.columns]
    print(f"Используется {len(available_features)} признаков из {len(base_features)}")
    
    X = df[available_features].copy()
    X = X.fillna(0)
    
    return X, available_features

def apply_model_to_vladimir():

    print("ПРИМЕНЕНИЕ МОДЕЛИ К ВЛАДИМИРСКОЙ ОБЛАСТИ")
    print("=" * 80)
    print("ВАЖНО: Модель предсказывает ОТНОСИТЕЛЬНУЮ ЗАСЕЛЕННОСТЬ зданий")
    print("=" * 80)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("\n1. ЗАГРУЗКА ДАННЫХ ВЛАДИМИРСКОЙ ОБЛАСТИ")
    df_vladimir = load_and_prepare_vladimir_data()
    
    if df_vladimir is None or len(df_vladimir) == 0:
        print("ОШИБКА: Не удалось загрузить данные")
        return
    
    print("\n2. СОЗДАНИЕ ПРИЗНАКОВ")
    df_vladimir = create_features_for_vladimir(df_vladimir)
    
    print("\n3. ПОДГОТОВКА ДЛЯ ПРЕДСКАЗАНИЯ")
    X, features = get_features_for_prediction(df_vladimir)
    
    print("\n4. ЗАГРУЗКА МОДЕЛИ")
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print(f"Модель загружена: {MODEL_PATH}")
        print(f"Scaler загружен: {SCALER_PATH}")
    except Exception as e:
        print(f"ОШИБКА загрузки модели: {e}")
        return
    
    print("\n5. МАСШТАБИРОВАНИЕ ПРИЗНАКОВ")
    X_scaled = scaler.transform(X)
    
    print("\n6. ПРЕДСКАЗАНИЕ ОТНОСИТЕЛЬНОЙ ЗАСЕЛЕННОСТИ")
    y_pred = model.predict(X_scaled)
    
    y_pred = np.maximum(y_pred, 0.1)
    
    # Нормализуем предсказания в диапазоне 0-1 для относительной оценки
    y_pred_normalized = (y_pred - y_pred.min()) / (y_pred.max() - y_pred.min())
    
    df_vladimir['relative_population_score'] = y_pred_normalized
    df_vladimir['predicted_population_raw'] = y_pred
    df_vladimir['population_category'] = pd.qcut(y_pred_normalized, q=5, 
                                                 labels=['очень низкая', 'низкая', 'средняя', 'высокая', 'очень высокая'])
    
    print("\n7. АНАЛИЗ РЕЗУЛЬТАТОВ")
    
    print(f"  Количество зданий: {len(df_vladimir):,}")
    print(f"  Диапазон относительных оценок: {y_pred_normalized.min():.3f} - {y_pred_normalized.max():.3f}")
    print(f"  Средняя относительная оценка: {y_pred_normalized.mean():.3f}")
    print(f"  Медианная относительная оценка: {np.median(y_pred_normalized):.3f}")
    
    print(f"\nРаспределение по категориям заселенности:")
    for category, count in df_vladimir['population_category'].value_counts().items():
        percentage = count / len(df_vladimir) * 100
        print(f"  {category:15}: {count:6} зданий ({percentage:5.1f}%)")
    
    print(f"\nРаспределение по типам зданий:")
    if 'parsed_building_type' in df_vladimir.columns:
        for btype, group in df_vladimir.groupby('parsed_building_type'):
            count = len(group)
            avg_score = group['relative_population_score'].mean()
            print(f"  {btype:15}: {count:6} зданий, средняя оценка {avg_score:.3f}")
    
    city_report_file = os.path.join(OUTPUT_DIR, "vladimir_city_population_report.txt")
    
    if 'parsed_city' in df_vladimir.columns:
        print("\n" + "=" * 80)
        print("ОТЧЕТ ПО ОТНОСИТЕЛЬНОЙ ЗАСЕЛЕННОСТИ ПО ГОРОДАМ")
        print("=" * 80)
        print("ВАЖНО: Эти данные показывают ОТНОСИТЕЛЬНУЮ заселенность, не абсолютное население!")
        print("Более высокие значения = более плотная заселенность")
        print("=" * 80)
        
        city_stats = df_vladimir.groupby('parsed_city').agg({
            'relative_population_score': ['count', 'mean', 'median', 'sum', 'std'],
            'predicted_population_raw': ['mean', 'sum']
        }).round(3)
        
        city_stats.columns = ['количество_зданий', 'средняя_оценка', 'медианная_оценка', 
                            'сумма_оценок', 'std_оценки', 'среднее_сырое', 'сумма_сырое']
        
        city_stats['индекс_заселенности'] = (city_stats['средняя_оценка'] - city_stats['средняя_оценка'].min()) / \
                                           (city_stats['средняя_оценка'].max() - city_stats['средняя_оценка'].min())
        
        city_stats = city_stats.sort_values('индекс_заселенности', ascending=False)
        
        print(f"\n{'№':>3} {'ГОРОД':25} {'ЗДАНИЙ':>8} {'ИНДЕКС':>8} {'СРЕДНЯЯ':>8} {'МЕДИАНА':>8}")
        print("-" * 78)
        
        for i, (city, row) in enumerate(city_stats.iterrows(), 1):
            buildings = int(row['количество_зданий'])
            index = row['индекс_заселенности']
            avg_score = row['средняя_оценка']
            median_score = row['медианная_оценка']
            
            if i <= 20:
                city_display = city[:23] + "..." if len(city) > 23 else city
                print(f"{i:>3} {city_display:25} {buildings:>8} {index:>8.3f} {avg_score:>8.3f} {median_score:>8.3f}")
        
        print("-" * 78)
        
        report_lines = []
        report_lines.append("ОТЧЕТ ПО ОТНОСИТЕЛЬНОЙ ЗАСЕЛЕННОСТИ ГОРОДОВ ВЛАДИМИРСКОЙ ОБЛАСТИ")
        report_lines.append("ВАЖНО: Эти данные показывают ОТНОСИТЕЛЬНУЮ заселенность, не абсолютное население!")
        report_lines.append("Более высокие значения = более плотная заселенность")
        report_lines.append("")
        
        report_lines.append(f"{'№':>3} {'ГОРОД':30} {'ЗДАНИЙ':>8} {'ИНДЕКС':>8} {'СРЕДНЯЯ':>8} {'МЕДИАНА':>8}")
        report_lines.append("-" * 85)
        
        for i, (city, row) in enumerate(city_stats.iterrows(), 1):
            buildings = int(row['количество_зданий'])
            index = row['индекс_заселенности']
            avg_score = row['средняя_оценка']
            median_score = row['медианная_оценка']
            
            city_display = city[:28] + "..." if len(city) > 28 else city
            report_lines.append(f"{i:>3} {city_display:30} {buildings:>8} {index:>8.3f} {avg_score:>8.3f} {median_score:>8.3f}")
        
        report_lines.append("")
        report_lines.append("ИНТЕРПРЕТАЦИЯ РЕЗУЛЬТАТОВ:")
        report_lines.append("1. Индекс заселенности: 0 = наименьшая заселенность, 1 = наибольшая")
        report_lines.append("2. Средняя оценка: среднее значение относительной заселенности зданий в городе")
        report_lines.append("3. Медиана: медианное значение относительной заселенности")
        report_lines.append("")
        
        with open(city_report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"\n  Подробный отчет сохранен в: {city_report_file}")
        
        city_excel_file = os.path.join(OUTPUT_DIR, "vladimir_city_relative_statistics.xlsx")
        city_stats.to_excel(city_excel_file)
        print(f"  Статистика по городам сохранена в: {city_excel_file}")
    
    
    result_columns = [
        'house_id', 'lat', 'lon', 'parsed_region', 'parsed_city',
        'parsed_street', 'parsed_house', 'building_type', 'building_levels',
        'parsed_building_type', 'settlement_type',
        'relative_population_score', 'predicted_population_raw', 'population_category'
    ]
    
    available_columns = [col for col in result_columns if col in df_vladimir.columns]
    df_results = df_vladimir[available_columns].copy()
    
    df_results.to_excel(VLADIMIR_RESULTS, index=False)
    print(f"  Результаты сохранены в: {VLADIMIR_RESULTS}")
    
    print(f"\nОСНОВНЫЕ ВЫВОДЫ:")
    print(f"• Анализировано зданий: {len(df_vladimir):,}")
    print(f"• Метод: Оценка относительной заселенности (не абсолютной численности)")
    print(f"• Диапазон оценок: {y_pred_normalized.min():.3f} - {y_pred_normalized.max():.3f}")
    
    print(f"\nОГРАНИЧЕНИЯ:")
    print("• Не дает абсолютной численности населения")
    print("• Показывает только относительные различия")
    print("• Требует калибровки для перевода в абсолютные значения")
    
    print(f"\nСОЗДАННЫЕ ФАЙЛЫ:")
    print(f"  • Результаты по зданиям: {VLADIMIR_RESULTS}")
    print(f"  • Отчет по городам: {city_report_file}")
    print(f"  • Статистика: {city_excel_file}")
    
    return df_vladimir

def main():

    print("ПРИМЕНЕНИЕ МОДЕЛИ ОЦЕНКИ НАСЕЛЕНИЯ К ВЛАДИМИРСКОЙ ОБЛАСТИ")
    
    df_results = apply_model_to_vladimir()
    
    if df_results is not None:
        print("МОДЕЛЬ УСПЕШНО ПРИМЕНЕНА!")

if __name__ == "__main__":
    main()