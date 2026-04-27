# Stáž - úkol 

## Scénář:

Dostal/a jste 2 roky denních dat o prodejích Japonské [matchi](https://en.wikipedia.org/wiki/Matcha). Prodeje závisí na dni, ceně, kterou jsme účtovali, na tom, zda běžela promo akce, a na dni v týdnu. Vaším úkolem je porozumět struktuře dat, postavit model, který predikuje prodeje z dostupných featur, a vytvořit předpověď na dalších 90 dní - je na vás, zda k úkolu přistoupíte spíše koncepčně nebo prakticky a programátorsky. 

Chceme ale v každém případě vidět alespoň část vámi psaného kódu, ideálně vč. enkapsulace do class(es) a využití nějaké machine learning knihovny.

Pokuste se vzít zadaná data a zamyslet se nad úkolem. Cílem není mít perfektní kód, ale je to způsob, jakým my můžeme zhodnotit Váš způsob přemýšlení, přístup k zadaným úkolům, schopnost pracovat (částečně) samostně, apod. 

Přesnou interpretaci úkolu necháme na vás, může to být např., ale ne nutně:
 - EDA
 - feature engineering
 - vizualizace dat
 - tvorba modelu
 - porovnávání více modelů
 - teoretický návrh metodiky
 - praktická implementace
 - apod.


## Toolchain
Programujte v Pythonu, můžete volně používat zdroje na internetu. Částečně smíte používat LLMka (a je to do rozumné míry i žádoucí, abyste se zbytečně neznevýhodnili), nicméně byste měli všemu, co nám submitnete, detailně rozumět, včetně jakéhokoliv AI generated kódu. Ideálně AI kód i označte, například tagem <LLM> /code </LLM> 


## Data

Dvě .csv v `data/`:

- `train.csv` — sloupce: `date`, `t`, `dow`, `price`, `promo`, `sales`
- `test.csv`  — sloupce stejné **ale bez `sales` **

Význam sloupců:

| sloupec  | popis                                              |
|---------|----------------------------------------------------------|
| `date`  | kalendářní datum                                            |
| `t`     | den index |
| `dow`   | den v týdnu, 0 = Pondělí … 6 = Nedělě                     |
| `price` | cena našeho produktu ten den (spojitá)|
| `promo` | 1 pokud jsme dělali promo akci, jinak 0            |
| `sales` | jednotek prodáno ten den - náš target                             |

## Dodáte

Zip / repo obsahující:

- kód
- `predictions.csv`
- váš komentář úkolu, kde popisujete svůj myšlenkový proces, co jste stihli udělat, co jste nestihli udělat, ale chtěli jste (chápeme, že úkolu nemůžete věnovat 3 dny času..) - může to být cokoliv od komentářů v kódu nebo Jupyter notebooku, prezentace, readme, Excalidraw diagramu,.. Opět zdůrazňuji, že nám jde hlavně především o to, jak přemýšlíte, jak dokážete řešit open ended problémy, zda umíte zvolit vhodné technické řešení nebo přijít s přístupem, který k nalezení takového řešení povede


Good luck