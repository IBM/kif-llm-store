# Copyright (C) 2024 IBM Corp.
# SPDX-License-Identifier: Apache-2.0

import httpx
from typing_extensions import Iterator, override

from ..context_generator import ContextGenerator


class ExchangePlugin(
    ContextGenerator.Plugin,
    plugin_name='exchange',
    plugin_patterns=['.'],
    plugin_priority=101,
):

    _exchange_codes = {
        "AMS": "Amsterdam Exchange",
        "ASE": "American Stock Exchange",
        "ASX": "Australian Stock Exchange",
        "ATH": "Athens Stock Exchange",
        "B3 (stock exchange)": "São Paulo Stock Exchange",
        "B3": "São Paulo Stock Exchange",
        "BDP": "Budapest Stock Exchange",
        "BER": "Berlin Stock Exchange",
        "BKK": "Thailand Stock Exchange",
        "BMF": "Brazil Mercantile & Futures Exchange",
        "BRU": "Euronext Brussels",
        "BSE": "Bombay Stock Exchange",
        "BTS": "BATS Global Markets",
        "BUE": "Buenos Aires Stock Exchange",
        "BUL": "Bulgarian Stock Exchange",
        "BVC": "Colombia Stock Exchange",
        "CAI": "Egyptian Exchange",
        "CNSX": "Canadian Securities Exchange",
        "CPH": "Copenhagen Stock Exchange",
        "CSE": "Copenhagen Stock Exchange",
        "CYP": "Cyprus Stock Exchange",
        "DFM": "Dubai Financial Market",
        "DUB": "Irish Stock Exchange",
        "DUS": "Dusseldorf Exchange",
        "ETO": "London Stock Exchange",
        "FRA": "Frankfurt Stock Exchange",
        "FSE": "Fukuoka Stock Exchange",
        "KRX": "Korean Stock Exchange",
        "GER": "XETRA Stock Exchange",
        "HAM": "Hamburg Stock Exchange",
        "HAN": "Hanover Stock Exchange",
        "HEL": "Helsinki Stock Exchange",
        "HKEX": "Hong Kong Exchanges And Clearing Ltd",
        "HKG": "Hong Kong Exchanges And Clearing Ltd",
        "港交所": "Hong Kong Exchanges And Clearing Ltd",
        "HSE": "Helsinki Stock Exchange",
        "HSX": "HoChiMinh Stock Exchange",
        "ICSE": "Iceland Stock Exchange",
        "IDK": "Indonesia Stock Exchange",
        "IEX": "Investor's Exchange, LLC",
        "IOB": "International Order Book",
        "ISE": "Italian Stock Exchange",
        "ISM": "Iceland Stock Exchange",
        "IST": "Istanbul Stock Exchange",
        "JKT": "Jakarta Stock Exchange",
        "JPX": "Tokyo Stock Exchange",
        "JSD": "JASDAQ Securities Exchange",
        "KAR": "Karachi Stock Exchange",
        "KLS": "Bursa Malaysia",
        "KLSE": "Bursa Malaysia",
        "LBC": "Casablanca Stock Exchange",
        "LSE": "London Stock Exchange",
        "LUX": "Luxembourg Stock Exchange",
        "MAC": "Macedonian Stock Exchange",
        "MCE": "Madrid Stock Exchange (Bolsa de Madrid)",
        "MEX": "Mexican Stock Exchange",
        "MIL": "Borsa Italiana (Milan Stock Exchange)",
        "MUN": "Munich Exchange",
        "NAG": "Nagoya Stock Exchange",
        "NAS": "Nasdaq",
        "NASDAQ": "Nasdaq",
        "NCM": "Nasdaq",
        "NEO": "NEO Stock Exchange",
        "NGM": "Nordic Growth Market",
        "NMS": "NASDAQ Stock Market",
        "NSE": "National Stock Exchange of India",
        "NYQ": "New York Stock Exchange",
        "NYSE": "New York Stock Exchange",
        "NZE": "New Zealand Exchange",
        "OBB": "OTC Bulletin Board Market",
        "OPR": "OPRA",
        "OSE": "Osaka Securities Exchange",
        "OSL": "Oslo Stock Exchange",
        "PAR": "Euronext Paris",
        "PNK": "Pink Sheets",
        "PRG": "Prague Stock Exchange",
        "RSE": "OMX Baltic",
        "SAO": "São Paulo Stock Exchange",
        "SAU": "Saudi Stock Exchange",
        "SET": "Stock Exchange of Thailand",
        "SGO": "Santiago Stock Exchange",
        "SGX": "Singapore Exchange",
        "SHE": "Shenzhen Stock Exchange",
        "SNP": "S&P Indexes",
        "SSE": "Shanghai Stock Exchange",
        "STO": "Stockholm Stock Exchange",
        "STU": "Stuttgart Stock Exchange",
        "TAI": "Taiwan Stock Exchange",
        "TAL": "OMX Baltic Exchange - Talinn",
        "TASE": "Tel Aviv Stock Exchange",
        "TLV": "Tel Aviv Stock Exchange",
        "TOR": "Toronto Stock Exchange",
        "TPE": "Taiwan Stock Exchange",
        "TSE": "Tokyo Stock Exchange",
        "VAN": "Toronto Venture Exchange",
        "VIE": "Vienna Stock Exchange",
        "VSE": "OMX Baltic Exchange - Vilnius",
        "WAR": "Warsaw Stock Exchange",
        "XETRA": "Deutsche Börse Xetra",
        "ZSE": "Zagreb Stock Exchange",
        "ZUR": "SIX Swiss Exchange",
    }


class Yahoo_ExchangePlugin(
    ExchangePlugin,
    plugin_name='yahoo-finance-exchange',
    plugin_patterns=[r'.*https?://finance\.yahoo\.com/lookup/all\?s='],
    plugin_priority=101,
):
    """Plugin to get the exchange a company trades from Yahoo."""

    @override
    def _process(self, response: httpx.Response) -> Iterator[str]:
        import bs4

        soup = bs4.BeautifulSoup(response.text, features='html.parser')

        exchange_info = soup.find('span', string='Exchange')
        if exchange_info:
            exchange = (
                exchange_info.find_next('tr').find_all('td')[-1].get_text()
            )
            exchange = exchange.strip()
            sentence = f'The code of the exchange on which the {{subject}} company trades is {exchange}.'
            if self._exchange_codes.get(exchange):
                exchange = self._exchange_codes.get(exchange)
                sentence = f'{{subject}} shares can be traded at {exchange}.'
            yield sentence


class Wikipedia_ExchangePlugin(
    ExchangePlugin,
    plugin_name='company-exchange',
    plugin_patterns=[
        r'.*https?://(ja|ca|en|it|fr|th|ru|nl|fa|he|pt|zh)\.wikipedia\.org/wiki/'
    ],
    plugin_priority=101,
):
    """Plugin to get the exchange of a company from Wikipedia."""

    @override
    def _process(self, response: httpx.Response) -> Iterator[str]:
        import bs4

        soup = bs4.BeautifulSoup(response.content, features='html.parser')
        infobox = (
            soup.find('table', {'class': 'infobox'})
            if soup.find('table', {'class': 'infobox'})
            else soup.find('div', {'class': 'infobox_v2'})
        )

        protocol_end = self.url.find("://") + 3
        lang_code_end = self.url.find(".", protocol_end)
        language_code = self.url[protocol_end:lang_code_end]

        exchange_names = []

        if infobox:
            for row in infobox.find_all('tr'):
                header = row.find('th')
                try:
                    if header.find("a"):
                        if (
                            header.find("a").get('title') == 'Ticker symbol'
                            or header.text == 'Trading symbol'
                            or header.text == 'Company type'
                        ):
                            ticker_symbol = True
                    elif (
                        header.text == 'Trading symbol'
                        or header.text == 'Company type'
                    ):
                        ticker_symbol = True
                    else:
                        ticker_symbol = False
                except:
                    ticker_symbol = False
                if header and (
                    ticker_symbol
                    or 'Traded as' in header.get_text()
                    or 'Borse valori' in header.get_text()
                    or 'Borsa de cotització' in header.get_text()
                    or '市場情報' in header.get_text()
                    or 'Action' in header.get_text()
                    or 'การซื้อขาย' in header.get_text()
                    or 'نماد سهام شرکت' in header.get_text()
                    or 'בורסה' in header.get_text()
                    or 'Beurs' in header.get_text()
                    or 'Quotação' in header.get_text()
                    or '市場情報' in header.get_text()
                ):
                    cells = row.find_all('td')
                    for cell in cells:
                        a_tags = cell.find_all('a', title=True)
                        for a in a_tags:
                            if header.text != 'Trading symbol':
                                if a.next_sibling and ':' in a.next_sibling:
                                    if language_code == "en":
                                        if self._exchange_codes.get(
                                            a['title']
                                        ):
                                            exchange_names.append(
                                                self._exchange_codes.get(
                                                    a['title']
                                                )
                                            )
                                        if (
                                            "stock" in str(a['title']).lower()
                                            or "exchange"
                                            in str(a['title']).lower()
                                            or "Nasdaq" in a['title']
                                        ):
                                            exchange_names.append(a['title'])
                                    else:
                                        exchange_name = (
                                            self._exchange_codes.get(a.text)
                                        )
                                        if (
                                            "stock" in exchange_name.lower()
                                            or "exchange"
                                            in exchange_name.lower()
                                            or "Nasdaq" in exchange_name
                                        ):
                                            exchange_names.append(
                                                exchange_name
                                            )
                                elif a.next_sibling and "(" in a.next_sibling:
                                    exchange_name = (
                                        a.next_sibling.strip().strip("()")
                                    )
                                    acronym_value = self._exchange_codes.get(
                                        exchange_name
                                    )
                                    if acronym_value:
                                        exchange_names.append(acronym_value)
                                    elif (
                                        "stock" in exchange_name.lower()
                                        or "exchange" in exchange_name.lower()
                                        or "Nasdaq" in exchange_name
                                    ):
                                        exchange_names.append(exchange_name)
                                elif (
                                    row.find('td')
                                    and "Company type" in row.text
                                    and ":" in row.text
                                    and "(" in row.text
                                ):
                                    td = row.find('td')
                                    for link in td.find_all('a', title=True):
                                        exchange_names.append(link['title'])
                                elif (
                                    language_code == "he"
                                    or language_code == "pt"
                                    or language_code == "ja"
                                ):
                                    exchange_name = self._exchange_codes.get(
                                        a['title']
                                    )
                                    if exchange_name:
                                        exchange_names.append(exchange_name)
                                    else:
                                        exchange_names.append(a['title'])
                            else:
                                if (
                                    "stock" in str(a['title']).lower()
                                    or "exchange" in str(a['title']).lower()
                                    or a['title'] == "Nasdaq"
                                ):
                                    exchange_names.append(a['title'])
        else:
            if language_code == 'zh':
                body = soup.find('div', {'id': 'bodyContent'})
                for row in body.find_all('p'):
                    a_tags = row.find_all('a', title=True)
                    for a in a_tags:
                        exchange_name = self._exchange_codes.get(a.text)
                        if exchange_name:
                            exchange_names.append(exchange_name)
                        else:
                            exchange_names.append(a.text)

        if len(exchange_names) > 0:
            yield f"{{subject}} shares can be traded at {exchange_names}."

    def _extract_exchange(self, text: str, company_name: str) -> str:
        try:
            start_index = text.index(company_name)
            comma_index = text.index('，', start_index)
            colon_index = text.index('：', comma_index)
            exchange_info = text[comma_index + 1 : colon_index].strip()
            return exchange_info
        except:
            return ""


class Google_ExchangePlugin(
    ExchangePlugin,
    plugin_name='google',
    plugin_patterns=[r'.*https?://www\.google\.com/search\?q='],
    plugin_priority=101,
):
    """Plugin to get the exchange a company trades from Google."""

    @override
    def _process(self, response: httpx.Response) -> Iterator[str]:
        import bs4

        soup = bs4.BeautifulSoup(response.text, features='html.parser')

        info_box = soup.find("span", class_="FCUp0c rQMQod")
        if info_box:
            yield info_box.get_text()
