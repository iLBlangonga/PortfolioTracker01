##########################################
# portfolio_app.py
##########################################

import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, date
import math

import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Portfolio Tracker", layout="wide")

CSV_FILE = "transazioni.csv"
API_KEY = "66536c784fd3f8.36402127"

# Inizializziamo un reload_trigger
if "reload_trigger" not in st.session_state:
    st.session_state["reload_trigger"] = 0

@st.cache_data
def load_transactions(csv_path):
    df = pd.read_csv(csv_path, sep=",", parse_dates=["Datetime"], dayfirst=True)
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    df["Cost"]   = pd.to_numeric(df["Cost"], errors="coerce")
    df["Price"]  = pd.to_numeric(df["Price"], errors="coerce")
    for col in ["Ticker","Type","Sector","Subsector","Base"]:
        if col not in df.columns:
            df[col] = np.nan
    return df

def save_transactions(df, csv_path):
    df.to_csv(csv_path, index=False)

def download_eod_prices(ticker, start_date, end_date, api_key=API_KEY):
    base_url = "https://eodhistoricaldata.com/api/eod/"
    params = {
        "api_token": api_key,
        "from": str(start_date),
        "to": str(end_date),
        "fmt": "json"
    }
    url = f"{base_url}{ticker}"
    r = requests.get(url, params=params)
    if r.status_code == 200:
        data = r.json()
        if len(data) == 0:
            return pd.DataFrame(columns=[ticker])
        dfp = pd.DataFrame(data)
        dfp["date"] = pd.to_datetime(dfp["date"])
        dfp.set_index("date", inplace=True)
        dfp = dfp.sort_index()
        dfp = dfp[["close"]]
        dfp.columns = [ticker]
        return dfp
    else:
        return pd.DataFrame(columns=[ticker])

def compute_max_drawdown(aum_series: pd.Series) -> float:
    """Calcola il massimo drawdown (in percentuale) partendo da una serie AUM."""
    if len(aum_series) == 0:
        return np.nan
    peak = aum_series.cummax()
    drawdown = (aum_series - peak) / peak
    max_dd = drawdown.min()  # valore negativo
    return max_dd * 100  # in %

def calculate_portfolio(df_trx, user_end_date=None):
    outputs = {}
    if df_trx.empty:
        outputs["positions"]       = pd.DataFrame()
        outputs["df_prices_all"]   = pd.DataFrame()
        outputs["plot"]            = pd.DataFrame()
        outputs["monthly_returns"] = pd.Series(dtype=float)
        outputs["df_sector"]       = pd.DataFrame()
        outputs["df_subsector"]    = pd.DataFrame()
        outputs["largest_sector"]  = "N/A"
        outputs["annualized"]      = {}
        outputs["asset_tickers"]   = []
        outputs["cash_equiv"]      = []
        outputs["pos_shares"]      = {}
        outputs["pos_cost"]        = {}
        outputs["ticker_to_currency"] = {}
        return outputs

    df_trx = df_trx.sort_values("Datetime").reset_index(drop=True)
    start_date = df_trx["Datetime"].min().date()
    if user_end_date is not None:
        end_date = user_end_date
    else:
        end_date = df_trx["Datetime"].max().date()

    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    mask_assets = (df_trx["Ticker"].notnull()) & (~df_trx["Ticker"].str.contains(".FOREX"))
    tickers_info = df_trx.loc[mask_assets, ["Ticker","Base","Sector","Subsector"]].drop_duplicates()

    asset_tickers = tickers_info["Ticker"].unique()
    ticker_to_currency = dict(zip(tickers_info["Ticker"], tickers_info["Base"]))
    ticker_to_sector   = dict(zip(tickers_info["Ticker"], tickers_info["Sector"]))
    ticker_to_subsector= dict(zip(tickers_info["Ticker"], tickers_info["Subsector"]))

    # Se la colonna Sector/Subsector manca per qualche ticker, avremo "Unknown"
    def get_sector_subsec(t):
        sec  = ticker_to_sector[t] if t in ticker_to_sector else "Unknown"
        subc = ticker_to_subsector[t] if t in ticker_to_subsector else "Unknown"
        return sec, subc

    cash_equiv = ["XEON.XETRA","EL4W.XETRA"]
    asset_tickers = [t for t in asset_tickers if t not in cash_equiv]

    # Scarico prezzi
    prices_dict = {}
    all_instruments = list(asset_tickers) + cash_equiv
    for t in all_instruments:
        df_p = download_eod_prices(t, start_date, end_date)
        prices_dict[t] = df_p

    df_prices_all = pd.DataFrame(index=dates)
    if len(prices_dict)>0:
        df_prices_all = pd.concat(prices_dict.values(), axis=1)
    df_prices_all = df_prices_all.reindex(dates)
    df_prices_all.ffill(inplace=True)

    usdeur = download_eod_prices("USDEUR.FOREX", start_date, end_date)
    usdeur = usdeur.reindex(dates)
    usdeur.ffill(inplace=True)

    columns = list(asset_tickers) + cash_equiv + ["CASH_EUR","CASH_USD"]
    df_positions = pd.DataFrame(index=dates, columns=columns, dtype=float)
    df_positions[:] = 0.0

    current_positions = {a: 0.0 for a in asset_tickers}
    current_positions.update({c: 0.0 for c in cash_equiv})
    CASH_EUR=0.0
    CASH_USD=0.0

    pos_shares = {t: 0.0 for t in (asset_tickers+cash_equiv)}
    pos_cost   = {t: 0.0 for t in (asset_tickers+cash_equiv)}

    transactions_by_day = df_trx.groupby(df_trx["Datetime"].dt.date)

    for day in dates:
        day_date = day.date()
        if day_date in transactions_by_day.groups:
            day_trx = transactions_by_day.get_group(day_date)
            for idx,row in day_trx.iterrows():
                ticker = row["Ticker"]
                amount = row["Amount"] if not pd.isna(row["Amount"]) else 0.0
                cost   = row["Cost"]   if not pd.isna(row["Cost"])   else 0.0
                ttype  = row["Type"]   if not pd.isna(row["Type"])   else ""
                price  = row["Price"]  if not pd.isna(row["Price"])  else 0.0

                if ttype=="Deposit":
                    CASH_EUR += cost
                else:
                    if ticker and ".FOREX" in ticker:
                        if cost<0:
                            CASH_EUR+=cost
                            CASH_USD+=amount
                        else:
                            CASH_USD-=amount
                            CASH_EUR+=cost
                    else:
                        if ticker in (asset_tickers+cash_equiv):
                            asset_cur = ticker_to_currency.get(ticker,"EUR")
                            if asset_cur=="EUR":
                                if amount>0:
                                    new_cost = -cost if cost<0 else cost
                                    old_shares=pos_shares[ticker]
                                    old_cost  =pos_cost[ticker]
                                    pos_cost[ticker]= old_cost+new_cost
                                    pos_shares[ticker]= old_shares+amount
                                elif amount<0:
                                    if pos_shares[ticker]!=0:
                                        avg_cost_local= pos_cost[ticker]/pos_shares[ticker]
                                        pos_cost[ticker]-= avg_cost_local*abs(amount)
                                    pos_shares[ticker]+= amount
                                CASH_EUR+=cost
                                current_positions[ticker]+=amount

                            else:
                                usd_spent=price*amount
                                if amount>0:
                                    old_shares= pos_shares[ticker]
                                    old_cost  = pos_cost[ticker]
                                    pos_cost[ticker]= old_cost+usd_spent
                                    pos_shares[ticker]= old_shares+amount
                                elif amount<0:
                                    if pos_shares[ticker]!=0:
                                        avg_cost_local=pos_cost[ticker]/pos_shares[ticker]
                                        pos_cost[ticker]-= avg_cost_local*abs(amount)
                                    pos_shares[ticker]+= amount
                                current_positions[ticker]+=amount
                                CASH_USD-=usd_spent

        for a in asset_tickers+cash_equiv:
            df_positions.at[day,a] = current_positions[a]
        df_positions.at[day,"CASH_EUR"]=CASH_EUR
        df_positions.at[day,"CASH_USD"]=CASH_USD

    # Calcolo AUM
    aum_series=[]
    for day in dates:
        day_val_eur = df_positions.loc[day,"CASH_EUR"]
        usd_val     = df_positions.loc[day,"CASH_USD"]
        day_usd_eur=1.0
        if "USDEUR.FOREX" in usdeur.columns and day in usdeur.index:
            val_fx= usdeur.loc[day,"USDEUR.FOREX"]
            if not pd.isna(val_fx):
                day_usd_eur= val_fx

        day_val_eur+= usd_val*day_usd_eur

        for c in cash_equiv:
            qty_ce= df_positions.loc[day,c]
            ce_price= df_prices_all.loc[day,c] if c in df_prices_all.columns else 0.0
            day_val_eur+= qty_ce*ce_price

        for t in asset_tickers:
            qty= df_positions.loc[day,t]
            if pd.isna(qty):
                qty=0.0
            asset_price= df_prices_all.loc[day,t] if t in df_prices_all.columns else 0.0
            asset_cur= ticker_to_currency.get(t,"EUR")
            if asset_cur=="USD":
                val_eur= qty*asset_price*day_usd_eur
            else:
                val_eur= qty*asset_price
            day_val_eur+= val_eur
        aum_series.append(day_val_eur)

    df_positions["AUM"] = aum_series
    if len(df_positions)>0:
        initial_value= df_positions["AUM"].iloc[0]
    else:
        initial_value=1.0
    df_positions["NAV"]= (df_positions["AUM"]/initial_value)*100
    df_plot= df_positions.copy()
    df_plot["Returns"] = df_plot["AUM"].pct_change()

    df_monthly = df_positions["AUM"].groupby([df_positions.index.year, df_positions.index.month]).last()
    monthly_returns= df_monthly.pct_change()*100
    monthly_returns= monthly_returns.dropna()
    # Uso l'indice come "categoria" con anno-mese
    monthly_returns.index= [f"{y}-{m:02d}" for y,m in monthly_returns.index]

    # max drawdown
    max_dd = compute_max_drawdown(df_plot["AUM"])

    last_day= df_positions.index[-1]
    day_usd_eur=1.0
    instrument_values={}

    # Se esiste USDEUR.FOREX
    if ("USDEUR.FOREX" in usdeur.columns) and (last_day in usdeur.index):
        val_fx=usdeur.loc[last_day,"USDEUR.FOREX"]
        if not pd.isna(val_fx):
            day_usd_eur=val_fx

    outputs["positions"]= df_positions
    outputs["plot"]     = df_plot
    outputs["df_prices_all"]= df_prices_all
    outputs["monthly_returns"]= monthly_returns

    # Sector e Subsector
    # creeremo df_sector e df_subsector con relative label
    # costruiamo l'esposizione
    mask_last = df_positions.loc[last_day, :]
    data_expo= []
    for t in asset_tickers+cash_equiv:
        qty= mask_last[t]
        if pd.isna(qty):
            qty=0
        if t in df_prices_all.columns:
            price_local= df_prices_all.loc[last_day,t]
        else:
            price_local=0.0
        val_eur= qty*price_local
        if t in ["CASH_EUR","CASH_USD"]: 
            # handled after
            pass

        data_expo.append((t, qty, val_eur))

    # Cash
    c_eur= mask_last["CASH_EUR"]
    c_usd= mask_last["CASH_USD"]*day_usd_eur
    data_expo.append(("CASH_EUR","", c_eur))
    data_expo.append(("CASH_USD","", c_usd))

    # Montiamo df_exposure
    df_expo= pd.DataFrame(data_expo, columns=["Ticker","Quantity","ValueEUR"])

    # Aggiungiamo Sector e Subsector se disponibili
    def get_sector_subsec(t):
        if t in ticker_to_sector: 
            sec= ticker_to_sector[t]
        else:
            sec= "Cash" if t.startswith("CASH") else "Unknown"
        if t in ticker_to_subsector:
            sub= ticker_to_subsector[t]
        else:
            sub= "Unknown"
        return sec, sub
    sector_list= []
    subsec_list= []
    for i, row in df_expo.iterrows():
        tk= row["Ticker"]
        if tk in ["CASH_EUR"]:
            sector_list.append("Cash"); subsec_list.append("EUR")
        elif tk in ["CASH_USD"]:
            sector_list.append("Cash"); subsec_list.append("USD")
        else:
            s, sb= get_sector_subsec(tk)
            if tk in cash_equiv:
                s="Cash"
                sb="Equiv"
            sector_list.append(s if s else "Unknown")
            subsec_list.append(sb if sb else "Unknown")
    df_expo["Sector"]= sector_list
    df_expo["Subsector"]= subsec_list

    df_sector = df_expo.groupby("Sector")["ValueEUR"].sum().reset_index()
    df_subsector= df_expo.groupby(["Sector","Subsector"])["ValueEUR"].sum().reset_index()

    if len(df_sector)>0:
        largest_sector= df_sector.loc[df_sector["ValueEUR"].idxmax(),"Sector"]
    else:
        largest_sector="N/A"

    # metriche annualizzate
    daily_returns= df_plot["Returns"].dropna()
    mean_daily_return= daily_returns.mean()
    daily_volatility= daily_returns.std()
    annualized_mean= mean_daily_return*252
    annualized_vol= daily_volatility*math.sqrt(252)
    annualized_sharpe= annualized_mean/annualized_vol if annualized_vol!=0 else np.nan

    outputs["df_sector"]       = df_sector
    outputs["df_subsector"]    = df_subsector
    outputs["largest_sector"]  = largest_sector
    outputs["annualized"] = {
        "mean_return": annualized_mean,
        "volatility": annualized_vol,
        "sharpe": annualized_sharpe,
        "max_drawdown": float(max_dd)  # in %
    }
    outputs["asset_tickers"]= asset_tickers
    outputs["cash_equiv"]   = cash_equiv
    outputs["pos_shares"]   = pos_shares
    outputs["pos_cost"]     = pos_cost
    outputs["ticker_to_currency"]= ticker_to_currency

    return outputs

def main():
    st.title("Portfolio Tracker - Gestione IFO 01")

    # Gestione end_date e session state
    col_spacer, col_end, col_buttons= st.columns([5,2,2])

    with col_end:
        st.write("**Data di Fine**")
        user_end_date= st.date_input("Seleziona End Date", datetime.today())

    with col_buttons:
        if st.button("Usa data di oggi"):
            user_end_date= datetime.today().date()
            st.session_state["reload_trigger"] += 1

        if st.button("Riscarica dati"):
            st.session_state["reload_trigger"] += 1

    df_trx = load_transactions(CSV_FILE)
    st.subheader("Transazioni Correnti")
    st.dataframe(df_trx)

    st.subheader("Aggiungi / Modifica Transazione")
    with st.form("new_trx_form"):
        col1,col2,col3= st.columns(3)
        with col1:
            new_date   = st.date_input("Data", datetime.today())
            new_ticker = st.text_input("Ticker","XYZ.MI")
            new_type   = st.selectbox("Tipo",["BUY","SELL","Deposit","Withdraw"])
        with col2:
            new_amount = st.number_input("Amount (quote)",0.0)
            new_cost   = st.number_input("Cost", 0.0)
            new_price  = st.number_input("Price",0.0)
        with col3:
            new_sector     = st.text_input("Sector","Unknown")
            new_subsector  = st.text_input("Subsector","Unknown")
            new_base       = st.selectbox("Valuta Base",["EUR","USD"])

        submitted= st.form_submit_button("Salva Transazione")
        if submitted:
            new_row = {
                "Datetime": pd.to_datetime(new_date),
                "Ticker": new_ticker,
                "Type": new_type,
                "Amount": new_amount,
                "Cost": new_cost,
                "Price": new_price,
                "Sector": new_sector,
                "Subsector": new_subsector,
                "Base": new_base
            }
            df_trx = df_trx.append(new_row, ignore_index=True)
            save_transactions(df_trx, CSV_FILE)
            st.success("Transazione salvata!")
            st.session_state["reload_trigger"] += 1

    st.subheader("Elimina Transazione")
    if len(df_trx)>0:
        delete_index= st.number_input("Indice transazione", min_value=0, max_value=len(df_trx)-1, step=1)
        if st.button("Elimina"):
            df_trx.drop(df_trx.index[delete_index], inplace=True)
            save_transactions(df_trx, CSV_FILE)
            st.warning(f"Transazione {delete_index} eliminata.")
            st.session_state["reload_trigger"] += 1

    st.header("Analisi del Portafoglio")

    if len(df_trx)==0:
        st.info("Nessuna transazione!")
        return

    results= calculate_portfolio(df_trx, user_end_date=user_end_date)
    df_positions  = results["positions"]
    df_plot       = results["plot"]
    df_prices_all = results["df_prices_all"]
    monthly_returns= results["monthly_returns"]
    df_sector     = results["df_sector"]
    df_subsector  = results["df_subsector"]
    largest_sector= results["largest_sector"]
    annualized    = results["annualized"]
    asset_tickers = results["asset_tickers"]
    cash_equiv    = results["cash_equiv"]
    pos_shares    = results["pos_shares"]
    pos_cost      = results["pos_cost"]
    ticker_to_currency= results["ticker_to_currency"]

    if df_positions.empty:
        st.info("Nessuna posizione calcolabile con la data selezionata.")
        return

    last_day= df_positions.index[-1]

    # Tabellina separata per AUM e NAV
    st.subheader("AUM e NAV (tabella separata)")
    aum_nav_df= df_positions.loc[[last_day], ["AUM","NAV"]]
    st.dataframe(aum_nav_df)

    st.subheader(f"Posizioni Finali (ultimo giorno: {last_day.date()}) (Esclude Quantità Zero)")
    # Creiamo una tabella con i ticker che hanno shares>0
    final_table= []
    final_val_eur= df_positions.loc[last_day,"AUM"]
    for tk in asset_tickers+cash_equiv:
        shares= pos_shares.get(tk,0.0)
        if shares>0:
            # Calcoliamo la % su AUM (nessun valore assoluto)
            price_local=0.0
            if tk in df_prices_all.columns:
                price_local= df_prices_all.loc[last_day, tk]
            value_eur= shares*price_local
            pct= (value_eur/final_val_eur)*100 if final_val_eur>0 else 0.0
            final_table.append((tk, f"{pct:.2f}%"))

    # Ordina la tabella in base al ticker
    df_final_table= pd.DataFrame(final_table, columns=["Ticker","% su AUM"])
    st.dataframe(df_final_table)

    st.subheader("Grafici Plotly - AUM e NAV separati")

    # AUM plot
    fig_aum = go.Figure()
    fig_aum.add_trace(go.Scatter(x=df_plot.index, y=df_plot["AUM"], name="AUM(EUR)", line=dict(color='blue')))
    # dimensioni (width ridotto, height aumentato)
    fig_aum.update_layout(width=400, height=600, title="AUM Separato")
    st.plotly_chart(fig_aum, use_container_width=False)

    # NAV plot
    fig_nav = go.Figure()
    fig_nav.add_trace(go.Scatter(x=df_plot.index, y=df_plot["NAV"], name="NAV (base100)", line=dict(color='orange')))
    fig_nav.update_layout(width=400, height=600, title="NAV Separato")
    st.plotly_chart(fig_nav, use_container_width=False)

    # Ritorni Mensili con asse X "category"
    st.subheader("Ritorni Mensili (Bar) - Asse X solo per Mese")
    if len(monthly_returns)>0:
        fig_m = px.bar(
            x=monthly_returns.index, 
            y=monthly_returns.values,
            labels={"x":"Mese","y":"Ritorno (%)"},
            title="Ritorni Mensili",
            width=400, height=300
        )
        # forziamo x come category
        fig_m.update_layout(xaxis={'type':'category'})
        st.plotly_chart(fig_m, use_container_width=False)
    else:
        st.info("Nessun ritorno mensile disponibile.")

    st.subheader("Esposizione per Settore - Plotly")
    df_sector_sorted = df_sector.sort_values("ValueEUR", ascending=False)
    st.dataframe(df_sector_sorted)

    fig_sector= px.bar(
        df_sector_sorted,
        x="Sector", y="ValueEUR",
        title="Esposizione Settore",
        width=400, height=300
    )
    # for clarity
    st.plotly_chart(fig_sector, use_container_width=False)

    st.subheader("Esposizione per Subsettore - Plotly")
    df_subsector["Label"]= df_subsector["Sector"]+" - "+df_subsector["Subsector"]
    df_subs_sorted= df_subsector.sort_values("ValueEUR", ascending=False)
    fig_subs= px.bar(
        df_subs_sorted, 
        x="Label", y="ValueEUR",
        title="Esposizione Subsettore",
        width=400, height=300
    )
    fig_subs.update_xaxes(tickangle=45)
    st.plotly_chart(fig_subs, use_container_width=False)

    # Grafico della liquidità (Cash%) nel tempo
    st.subheader("Liquidità (in %) nel tempo - Plotly")
    # Calcolo la % di cassa daily
    # cassa = df_positions["CASH_EUR"] + df_positions["CASH_USD"] * day_usd_eur + eventuali cash_equiv
    # Ma day_usd_eur daily e i cash_equiv daily:
    # per semplicità, approssimo come facevamo nel "nel tempo".
    # creo una serie daily "cash_perc"
    df_liquidity= df_plot.copy()
    # Calcoliamo cash in EUR (approssimato) come:
    # cash_in_eur = CASH_EUR + CASH_USD * daily_usdeur + sum(cash_equiv)
    # costruiamo una serie daily
    daily_liq=[]
    for day in df_plot.index:
        c_eur= df_plot.loc[day,"CASH_EUR"]
        c_usd= df_plot.loc[day,"CASH_USD"]
        usd_eur= 1.0
        # se esiste usdeur
        if day in results["df_prices_all"].index and "USDEUR.FOREX" in results["df_prices_all"].columns:
            val_fx= results["df_prices_all"].loc[day,"USDEUR.FOREX"]
            if not pd.isna(val_fx):
                usd_eur= val_fx
        c_eur += c_usd * usd_eur

        # aggiungo i cash_equiv
        for c in cash_equiv:
            qty= df_plot.loc[day,c]
            ce_price= 0.0
            if c in df_prices_all.columns:
                ce_price= df_prices_all.loc[day,c]
            c_eur+= qty*ce_price

        aum= df_plot.loc[day,"AUM"]
        if aum>0:
            daily_liq.append((day, (c_eur/aum)*100))
        else:
            daily_liq.append((day,0))

    df_liq= pd.DataFrame(daily_liq, columns=["Date","Cash%"])
    df_liq.set_index("Date", inplace=True)

    fig_liq= go.Figure()
    fig_liq.add_trace(go.Scatter(x=df_liq.index, y=df_liq["Cash%"], mode="lines", name="Liquidità (%)"))
    fig_liq.update_layout(width=400, height=300, title="Liquidità nel Tempo (%)")
    st.plotly_chart(fig_liq, use_container_width=False)

    # Metriche annualizzate + MaxDrawdown
    st.subheader("Metriche Annualizzate + MaxDrawdown")
    annual_mean = annualized["mean_return"]*100
    annual_vol  = annualized["volatility"]*100
    annual_sharpe= annualized["sharpe"]
    max_dd= annualized.get("max_drawdown", np.nan)  # in %

    st.write(f"Annualized Mean Return: {annual_mean:.2f}%")
    st.write(f"Annualized Volatility: {annual_vol:.2f}%")
    st.write(f"Sharpe Ratio: {annual_sharpe:.2f}")
    st.write(f"Max Drawdown: {max_dd:.2f}%")

    st.info(f"Settore principale: {largest_sector}")
    st.write("reload_trigger:", st.session_state["reload_trigger"])

if __name__=="__main__":
    main()
