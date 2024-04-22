from utils.DAM_utils import DataLoader, TPESearch, OnlineTraining
import os
os.environ['NUMEXPR_MAX_THREADS'] = '18'

def main():
    online=True

    datapath = 'data'
    savepath = 'data/search'
    dataloader = DataLoader(datapath, lagrange=[i for i in range(0, 7)], fcrange=range(1, 3))

    if not online:
        tpe_search = TPESearch(name='NL DAM', dataloader=dataloader, savepath=savepath)
        study = tpe_search.run(n_trials=5000, show_progress_bar=True, restart=True, n_startup_trials=250)
    else:
        online = OnlineTraining(name='NL DAM', dataloader=dataloader, savepath=savepath)
        online.run()

if __name__ == '__main__':
    main()