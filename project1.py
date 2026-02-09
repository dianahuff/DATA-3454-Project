import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import folium
import geodatasets
import seaborn as sns

operations = pd.read_csv('operations.csv')
weather = pd.read_csv('Summary of Weather.csv')
station_loc = pd.read_csv('Weather Station Locations.csv')

operations.info()
weather.info()
station_loc.head()

# weather stations with lat-long
weather_loc = weather.merge(station_loc, 
                            left_on='STA', 
                            right_on='WBAN', 
                            how='inner')

g_weather = gpd.GeoDataFrame(weather_loc)

# Making geographical points from lon/lat
g_weather['point'] = gpd.points_from_xy(
    x=g_weather['Longitude'],
    y=g_weather['Latitude'],
)
# geopandas needs an active geometry
g_weather.set_geometry('point')
g_weather.head()

g_weather.set_geometry('point', inplace=True)

## Cleanup
# NOTE:
# takeoff lat/lon only included for ~1.7% of operations 
# shouldn't value it too much.

# have some country names here to clean up
# we only lose 2 rows tho, not bad. 2981 -> 2979
operations['Takeoff Latitude'] = pd.to_numeric(operations['Takeoff Latitude'], 
                                               errors='coerce')
# lose 1
operations['Takeoff Longitude'] = pd.to_numeric(operations['Takeoff Longitude'], 
                                               errors='coerce')

# Transofmring operations to geo
g_operations = gpd.GeoDataFrame(operations)
g_operations['takeoff'] = gpd.points_from_xy(
    x=g_operations['Takeoff Longitude'],
    y=g_operations['Takeoff Latitude']
)
g_operations['target'] = gpd.points_from_xy(
    x=g_operations['Target Longitude'],
    y=g_operations['Target Latitude']
)
# NOTE :
# any GIS operations (like finding nearest point)
# depend on the current geometry.
# Setting this here, tho it needs to be changed for analysis 
# on the takeoff point (and changed back).
g_operations.set_geometry('target', inplace=True)
g_operations.info()

# not needed anymore
del operations
del weather
del station_loc


world_map = folium.Map()
g_operations[:10]

g_operations['theater'] = g_operations['Theater of Operations']
g_operations['theater'].value_counts()

euro_map = gpd.read_file(geodatasets.get_path('naturalearth.land'))

fig, ax = plt.subplots()
# Important for file size, repeat on all other axes made.
ax.set_rasterized(True)

ax.set_xlim(-20, 50)
ax.set_ylim(25, 75)
euro_map.plot(ax=ax)

g_operations.loc[g_operations['theater'] == 'ETO'].plot(
    ax=ax,
    color='red'
)
#fig.show(rasterize=True)
fig

scrap = pd.read_csv('operations.csv')
scrap.info()

scrap['Takeoff Latitude']
scrap.loc[scrap['Takeoff Latitude'] == 'NEW GUINEA']['Takeoff Longitude']




#===================================

# Convert dates to datetime for proper matching
g_operations['Mission Date'] = pd.to_datetime(g_operations['Mission Date'], errors='coerce')
g_weather['Date'] = pd.to_datetime(g_weather['Date'], errors='coerce')

# Define precision levels from most precise to least precise
precisions = [2, 1, 0, -1, -2, -3, -4]  # Most precise to least precise

def merge_with_rolling_precision(operations_df, weather_df, precisions):
    # Initialize result dataframe with original operations
    result_df = operations_df.copy()
    
    # Add placeholder columns for target weather
    result_df['target_Precip'] = np.nan
    result_df['target_MeanTemp'] = np.nan
    
    # Add placeholder columns for takeoff weather  
    result_df['takeoff_Precip'] = np.nan
    result_df['takeoff_MeanTemp'] = np.nan
    
    # Store original index mapping
    original_indices = result_df.index.tolist()
    
    print(f"Starting merge process...")
    print(f"Total operations: {len(result_df)}")
    
    for precision in precisions:
        print(f"\nTrying precision level: {precision}")
        
        # Get boolean masks for currently unmatched records
        target_unmatched_mask = result_df['target_Precip'].isna()
        takeoff_unmatched_mask = result_df['takeoff_Precip'].isna()
        
        # Process target weather
        if target_unmatched_mask.any():
            # Get subset of operations that still need target weather
            temp_ops = result_df[target_unmatched_mask].copy()
            
            # Add rounded coordinates for this precision level
            temp_ops['target_lat_rounded'] = temp_ops['Target Latitude'].round(precision)
            temp_ops['target_lon_rounded'] = temp_ops['Target Longitude'].round(precision)
            
            # Add rounded coordinates to weather data
            temp_weather = weather_df.copy()
            temp_weather['lat_rounded'] = temp_weather['Latitude'].round(precision)
            temp_weather['lon_rounded'] = temp_weather['Longitude'].round(precision)
            
            # Perform merge for unmatched target weather
            target_matches = temp_ops.reset_index().merge(
                temp_weather[['lat_rounded', 'lon_rounded', 'Date', 'Precip', 'MeanTemp']],
                left_on=['target_lat_rounded', 'target_lon_rounded', 'Mission Date'],
                right_on=['lat_rounded', 'lon_rounded', 'Date'],
                how='inner'
            )
            
            if len(target_matches) > 0:
                # Extract the original indices and update the main dataframe
                original_match_indices = target_matches['index'].values
                result_df.loc[original_match_indices, 'target_Precip'] = target_matches['Precip'].values
                result_df.loc[original_match_indices, 'target_MeanTemp'] = target_matches['MeanTemp'].values
            
            print(f"  Target weather - matched at precision {precision}: {len(target_matches)}")
        
        # Process takeoff weather
        if takeoff_unmatched_mask.any():
            # Get subset of operations that still need takeoff weather
            temp_ops = result_df[takeoff_unmatched_mask].copy()
            
            # Add rounded coordinates for this precision level
            temp_ops['takeoff_lat_rounded'] = temp_ops['Takeoff Latitude'].round(precision)
            temp_ops['takeoff_lon_rounded'] = temp_ops['Takeoff Longitude'].round(precision)
            
            # Add rounded coordinates to weather data
            temp_weather = weather_df.copy()
            temp_weather['lat_rounded'] = temp_weather['Latitude'].round(precision)
            temp_weather['lon_rounded'] = temp_weather['Longitude'].round(precision)
            
            # Perform merge for unmatched takeoff weather
            takeoff_matches = temp_ops.reset_index().merge(
                temp_weather[['lat_rounded', 'lon_rounded', 'Date', 'Precip', 'MeanTemp']],
                left_on=['takeoff_lat_rounded', 'takeoff_lon_rounded', 'Mission Date'],
                right_on=['lat_rounded', 'lon_rounded', 'Date'],
                how='inner'
            )
            
            if len(takeoff_matches) > 0:
                # Extract the original indices and update the main dataframe
                original_match_indices = takeoff_matches['index'].values
                result_df.loc[original_match_indices, 'takeoff_Precip'] = takeoff_matches['Precip'].values
                result_df.loc[original_match_indices, 'takeoff_MeanTemp'] = takeoff_matches['MeanTemp'].values
            
            print(f"  Takeoff weather - matched at precision {precision}: {len(takeoff_matches)}")
        
        # Count remaining unmatched
        remaining_target = result_df['target_Precip'].isna().sum()
        remaining_takeoff = result_df['takeoff_Precip'].isna().sum()
        
        print(f"  Remaining unmatched - Target: {remaining_target}, Takeoff: {remaining_takeoff}")
        
        # Early exit if all matches are found
        if remaining_target == 0 and remaining_takeoff == 0:
            print(f"  All matches found! Exiting precision loop early.")
            break
    
    return result_df

# Execute the rolling precision merge
final_df = merge_with_rolling_precision(g_operations, g_weather, precisions)

# Check results
print(f"\nFinal Results:")
print(f"Original operations: {len(g_operations)}")
print(f"Final combined: {len(final_df)}")
print(f"Target weather coverage: {final_df['target_Precip'].notna().sum()} / {len(final_df)} ({final_df['target_Precip'].notna().mean()*100:.1f}%)")
print(f"Takeoff weather coverage: {final_df['takeoff_Precip'].notna().sum()} / {len(final_df)} ({final_df['takeoff_Precip'].notna().mean()*100:.1f}%)")

# View sample
print("\nSample of merged data:")
print(final_df[['Mission Date', 'Target Latitude', 'Target Longitude',
                'target_Precip', 'target_MeanTemp', 
                'Takeoff Latitude', 'Takeoff Longitude',
                'takeoff_Precip', 'takeoff_MeanTemp']].head(10))

final_df.info()



def categorize_precip(value):
    if pd.isna(value):
        return 'Unknown'
    elif value == '0':
        return 'No Rain'
    elif value == 'T':
        return 'Trace'
    else:
        try:
            num_val = float(value)
            if num_val <= 1:
                return 'Light Rain'
            elif num_val <= 5:
                return 'Moderate Rain'
            else:
                return 'Heavy Rain'
        except:
            return 'Other'

# Apply categorization to the precipitation column
final_df['Precip_Category'] = final_df['target_Precip'].apply(categorize_precip)

# Filter for top aircraft and valid precipitation categories
top_aircraft = ['B24', 'B17', 'B25', 'A20', 'B26']
filtered_df = final_df[final_df['Aircraft Series'].isin(top_aircraft)]

# Create a cleaner heatmap
plt.figure(figsize=(10, 6))
cross_tab = pd.crosstab(filtered_df['Aircraft Series'], 
                       filtered_df['Precip_Category'])
sns.heatmap(cross_tab, annot=True, fmt='d', cmap='Blues')
plt.title('Aircraft Series vs Target Precipitation Conditions')
plt.tight_layout()
plt.show()

# Alternative: Grouped bar chart
cross_tab_pct = pd.crosstab(filtered_df['Aircraft Series'], 
                           filtered_df['Precip_Category'], 
                           normalize='index') * 100
ax = cross_tab_pct.plot(kind='bar', stacked=False, figsize=(12, 6))
plt.title('Percentage Distribution of Precipitation Conditions by Aircraft Type')
plt.ylabel('Percentage (%)')
plt.xticks(rotation=45)
plt.legend(title='Precipitation Category')
plt.tight_layout()
plt.show()





# Create numeric precipitation values
def precip_to_numeric(value):
    if pd.isna(value):
        return np.nan
    elif value == '0':
        return 0
    elif value == 'T':
        return 0.01  # Small trace amount
    else:
        try:
            return float(value)
        except:
            return np.nan

final_df['Precip_Numeric'] = final_df['target_Precip'].apply(precip_to_numeric)

# Filter out missing altitude values and set upper limit for better visualization
altitude_df = final_df.dropna(subset=['Altitude (Hundreds of Feet)', 'target_MeanTemp'])
altitude_df = altitude_df[altitude_df['Altitude (Hundreds of Feet)'] <= 350]  # Filter out outliers above 350

# Create scatter plot of bombing altitude vs mean temperature
plt.figure(figsize=(12, 8))
scatter = plt.scatter(altitude_df['target_MeanTemp'], 
                     altitude_df['Altitude (Hundreds of Feet)'], 
                     alpha=0.6, s=20)
plt.xlabel('Target Mean Temperature')
plt.ylabel('Bombing Altitude (Hundreds of Feet)')
plt.title('Bombing Altitude vs Target Mean Temperature (Filtered < 350)')
plt.ylim(0, 350)  # Set y-axis limit
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Filter for precipitation analysis as well
altitude_precip_df = final_df.dropna(subset=['Altitude (Hundreds of Feet)', 'Precip_Numeric'])
altitude_precip_df = altitude_precip_df[altitude_precip_df['Altitude (Hundreds of Feet)'] <= 350]

plt.figure(figsize=(12, 8))
scatter = plt.scatter(altitude_precip_df['Precip_Numeric'], 
                     altitude_precip_df['Altitude (Hundreds of Feet)'], 
                     alpha=0.6, s=20)
plt.xlabel('Target Precipitation Amount')
plt.ylabel('Bombing Altitude (Hundreds of Feet)')
plt.title('Bombing Altitude vs Target Precipitation Amount (Filtered < 350)')
plt.ylim(0, 350)  # Set y-axis limit
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Create box plots to show altitude distribution by precipitation category (filtered)
filtered_box_df = altitude_precip_df[altitude_precip_df['Altitude (Hundreds of Feet)'] <= 350]

plt.figure(figsize=(12, 8))
sns.boxplot(data=filtered_box_df, x='Precip_Category', y='Altitude (Hundreds of Feet)')
plt.xlabel('Precipitation Category')
plt.ylabel('Bombing Altitude (Hundreds of Feet)')
plt.title('Distribution of Bombing Altitude by Precipitation Category (Filtered < 350)')
plt.ylim(0, 350)  # Set y-axis limit
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Create a 2D histogram/heatmap for temperature vs altitude (filtered)
plt.figure(figsize=(10, 8))
hb = plt.hexbin(altitude_df['target_MeanTemp'], 
               altitude_df['Altitude (Hundreds of Feet)'], 
               gridsize=30, cmap='Blues')
plt.colorbar(hb)
plt.xlabel('Target Mean Temperature')
plt.ylabel('Bombing Altitude (Hundreds of Feet)')
plt.title('Heatmap: Bombing Altitude vs Target Mean Temperature (Filtered < 350)')
plt.ylim(0, 350)  # Set y-axis limit
plt.tight_layout()
plt.show()