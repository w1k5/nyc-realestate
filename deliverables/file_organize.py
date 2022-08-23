import pandas as pd

class dataFrameFinal():
    def gatherData():
        df_bronx = pd.read_csv("2021_bronx.csv", skiprows=6)
        df_brooklyn = pd.read_csv("2021_brooklyn.csv", skiprows=6)
        df_queens = pd.read_csv("2021_queens.csv", skiprows=6)
        df_manhattan = pd.read_csv("2021_manhattan.csv", skiprows=6)
        df_statenisland = pd.read_csv("2021_staten_island.csv", skiprows=6)

        frames = [df_queens, df_bronx, df_brooklyn, df_manhattan, df_statenisland]
        result = pd.concat(frames)

        #drop NaN and gifted properties
        result = result[result["SALE PRICE"].notna()]
        result.replace(',','', regex=True, inplace=True)
        result = result[result["SALE PRICE"] != "0"]
        result = result[result["GROSS \nSQUARE FEET"] != ""]
        result = result[result["GROSS \nSQUARE FEET"] != "0"]

        #add population density
        populations = pd.read_csv("nyc_zip_borough_neighborhoods_pop.csv")
        with_pop = pd.merge(left=populations, right=result, how='left', left_on='zip', right_on='ZIP CODE')

        #add data regarding estimated income
        incomes = pd.read_csv("Census Data_Income by Zip.csv", skiprows=1)
        incomes["Geographic Area Name"] = incomes["Geographic Area Name"].str.replace('ZCTA5 ', '').astype(int)
        incomes.replace(',','', regex=True, inplace=True)
        incomes = incomes[incomes["Estimate!!Households!!Median income (dollars)"] != "0"]

        with_income = pd.merge(left=incomes, right=with_pop, how='left', left_on='Geographic Area Name', right_on='ZIP CODE')

        #add data regarding race
        race = pd.read_csv("Census Data_Race by Zip.csv")
        race = race[race["Total:!!White alone"].str.contains("NaN")==False]
        race["Total:!!White alone"] = race["Total:!!White alone"].str.replace(',', '').astype(int)
        race["Total:"] = race["Total:"].str.replace(',', '').astype(int)
        race["PERCENT WHITE"] = 100*(race["Total:!!White alone"].div(race["Total:"].values))
        race["Label (Grouping)"] = race["Label (Grouping)"].str.replace('ZCTA5 ', '').astype(int)
        with_race = pd.merge(left=race, right=with_income, how='left', left_on='Label (Grouping)', right_on='ZIP CODE')
        dummies = pd.get_dummies(with_race["BOROUGH"])
        dummies.rename(columns = {4.0:'QUEENS', 2.0:'BRONX', 5.0:'STATEN_ISLAND', 3.0:'BROOKLYN', 1.0:'MANHATTAN'}, inplace = True)
        with_race = pd.concat([dummies, with_race], axis=1)

        #exclude unnecessary pieces of data
        wanted = ["COMMERCIAL\nUNITS", "RESIDENTIAL\nUNITS", "BUILDING CLASS CATEGORY", "PERCENT WHITE", "Estimate!!Households!!Median income (dollars)", 'ZIP CODE', 'QUEENS', 'MANHATTAN', 'STATEN_ISLAND', 'BRONX', 'BROOKLYN', 'NEIGHBORHOOD', 'ADDRESS', 'RESIDENTIAL\nUNITS', 'COMMERCIAL\nUNITS', 'LAND \nSQUARE FEET', 'SALE PRICE', 'density']
        with_race = with_race.loc[:, with_race.columns.intersection(wanted)]
        with_race = with_race[with_race["LAND \nSQUARE FEET"].str.contains("NaN")==False]
        with_race = with_race.reset_index(drop=True)

        #gather information regarding number of possible subways to be taken in the given neighborhood
        subways = pd.read_csv("NYC SUBWAYS.csv")
        subways["Services"] = subways["Services"].str.replace(' ', '')
        subways = subways[subways["Services"].str.contains("NaN")==False]
        subways["SERVICES_COUNT"] = subways["Services"].str.len()
        subwayCounter = subways.groupby("Neighborhood", as_index=False)[["SERVICES_COUNT"]].sum()
        subwayCounter["Neighborhood"] = subwayCounter["Neighborhood"].str.upper()

        with_race = pd.merge(left=with_race, right=subwayCounter, how='left', left_on="NEIGHBORHOOD", right_on='Neighborhood')
        with_race["SERVICES_COUNT"] = with_race["SERVICES_COUNT"].fillna(0)
        with_race = with_race.drop(columns="Neighborhood")

        #categorize type of location
        with_race["COOP"] = 1*with_race["BUILDING CLASS CATEGORY"].str.contains("10" or "09")
        with_race["CONDO"] = 1*with_race["BUILDING CLASS CATEGORY"].str.contains("CONDO")
        with_race["FAMILY_DWELLING"] = 1*with_race["BUILDING CLASS CATEGORY"].str.contains("FAMILY")
        with_race["ELEVATOR"] = 1*with_race["BUILDING CLASS CATEGORY"].str.contains("ELEVATOR")

        #number of units
        with_race.loc[with_race["RESIDENTIAL\nUNITS"]==0, ["RESIDENTIAL\nUNITS"]] = 1*with_race["BUILDING CLASS CATEGORY"].str.contains("APARTMENT")==0
        with_race.loc[with_race["RESIDENTIAL\nUNITS"].isna(), ["RESIDENTIAL\nUNITS"]] = 0
        with_race.loc[with_race["COMMERCIAL\nUNITS"].isna(), ["COMMERCIAL\nUNITS"]] = 0

        #drop columns
        with_race = with_race.drop(columns=['NEIGHBORHOOD', 'ADDRESS', 'ZIP CODE', 'BUILDING CLASS CATEGORY'])

        #organization
        with_race.rename(columns = {'density':'POPULATION_DENSITY', "Estimate!!Households!!Median income (dollars)":"MEDIAN_INCOME"}, inplace = True)
        for x in list(with_race):
            with_race[x] = pd.to_numeric(with_race[x], errors='coerce')
        with_race = with_race.dropna()
        with_race = with_race.rename(columns = {"COMMERCIAL\nUNITS":"COMMERCIAL_UNITS", "RESIDENTIAL\nUNITS":"RESIDENTIAL_UNITS", "BUILDING CLASS CATEGORY":"BUILDING_CLASS_CATEGORY", "PERCENT WHITE":"PERCENT_WHITE", 'LAND \nSQUARE FEET':'LAND_SQUARE_FEET', 'SALE PRICE':'SALE_PRICE'})

        with_race.to_csv("filename.csv", index=False)
        return with_race