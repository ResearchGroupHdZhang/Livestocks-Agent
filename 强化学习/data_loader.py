'''
data should store in ../data folder
'''

import pandas as pd
import os
import numpy as np

country_mapping = {
    'usa': '美国',
    'br': '巴西',
    'us': '欧盟',
    'cn': '中国',
    'aus': '澳大利亚'
}

def load_datas(country):
    country = country_mapping[country]
    # 读取数据
    Move_in = pd.read_excel(os.path.join(os.path.dirname(__file__), '..','data',country,f"{country}空间优化数据.xlsx"), engine='openpyxl', sheet_name='移入城市')
    Move_in = Move_in.dropna(axis=1, how='all')
    Move_in = Move_in.fillna(0)

    Move_out = pd.read_excel(os.path.join(os.path.dirname(__file__), '..','data',country,f"{country}空间优化数据.xlsx"), engine='openpyxl', sheet_name='移出城市')
    Move_out = Move_out.dropna(axis=1, how='all')
    Move_out = Move_out.fillna(0)

    # 用列最大值填充缺失值
    Move_out = Move_out.fillna(Move_out.max())
    # load move_in data
    ID_move_in = Move_in[['ID', 'city', 'county']]
    Amount_move_in = Move_in[[col for col in Move_in.columns if 'num' in col]]
    Amount_move_in = Amount_move_in.astype(np.int64)

    N_demand_move_in = Move_in.iloc[:, 12]
    ammonia_density_move_in = Move_in.iloc[:, 14]
    livestock_PB_move_in = Move_in.iloc[:, 17]
    senstivity_move_in = Move_in.iloc[:, -8]
    relative_pm25_move_in = Move_in.iloc[:, -7]
    Target_move_in = {'N_demand': N_demand_move_in, 'ammonia_density': ammonia_density_move_in, 'livestock_PB': livestock_PB_move_in, 'sensitivity': senstivity_move_in, 'relative_pm25': relative_pm25_move_in}
    
    N_demand_Coef_move_in = Move_in.iloc[:, 18:18+6]
    ammonia_density_Coef_move_in = Move_in.iloc[:, 24:24+6]
    livestock_PB_Coef_move_in = Move_in.iloc[:, -6:]
    Coef_move_in = {'N_demand': N_demand_Coef_move_in, 'ammonia_density': ammonia_density_Coef_move_in, 'livestock_PB': livestock_PB_Coef_move_in}
    
    # load move_out data
    ID_move_out = Move_out[['ID', 'city', 'county']]
    Amount_move_out = Move_out[[col for col in Move_out.columns if 'num' in col]]
    Amount_move_out = Amount_move_out.astype(np.int64)

    N_demand_move_out = Move_out.iloc[:, 12]
    ammonia_density_move_out = Move_out.iloc[:, 14]
    livestock_PB_move_out = Move_out.iloc[:, 17]
    sensitivity_move_out = Move_out.iloc[:, -8]
    relative_pm25_move_out = Move_out.iloc[:, -7]
    Target_move_out = {'N_demand': N_demand_move_out, 'ammonia_density': ammonia_density_move_out, 'livestock_PB': livestock_PB_move_out, 'sensitivity': sensitivity_move_out, 'relative_pm25': relative_pm25_move_out}
    
    N_demand_Coef_move_out = Move_out.iloc[:, 18:18+6]
    ammonia_density_Coef_move_out = Move_out.iloc[:, 24:24+6]
    livestock_PB_Coef_move_out = Move_out.iloc[:, -6:]
    Coef_move_out = {'N_demand': N_demand_Coef_move_out, 'ammonia_density': ammonia_density_Coef_move_out, 'livestock_PB': livestock_PB_Coef_move_out}

    return (ID_move_in, ID_move_out, Amount_move_in, Amount_move_out, Target_move_in, Coef_move_in, Target_move_out, Coef_move_out)

if __name__ == '__main__':
    ID_move_in, ID_move_out, Amount_move_in, Amount_move_out, Target_move_in, Coef_move_in, Target_move_out, Coef_move_out = load_datas('usa')
    print(ID_move_in)
    print(Amount_move_in)
    print(Amount_move_out)
    print(Target_move_in)
    print(Coef_move_in)
    print(Target_move_out)
    print(Coef_move_out)