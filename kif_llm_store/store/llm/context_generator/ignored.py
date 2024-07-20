# Copyright (C) 2024 IBM Corp.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Collection, Sequence, Set

from typing_extensions import Final

# ISO 3166-1, Alpha 2.
ISO3166_1_ALPHA2: Final[Set[str]] = {
    'AF',  # Afghanistan[b]
    'AX',  # Åland Islands
    'AL',  # Albania
    'DZ',  # Algeria
    'AS',  # American Samoa
    'AD',  # Andorra
    'AO',  # Angola
    'AI',  # Anguilla
    'AQ',  # Antarctica
    'AG',  # Antigua and Barbuda
    'AR',  # Argentina
    'AM',  # Armenia
    'AW',  # Aruba
    'AU',  # Australia
    'AT',  # Austria
    'AZ',  # Azerbaijan
    'BS',  # Bahamas
    'BH',  # Bahrain
    'BD',  # Bangladesh
    'BB',  # Barbados
    'BY',  # Belarus
    'BE',  # Belgium
    'BZ',  # Belize
    'BJ',  # Benin
    'BM',  # Bermuda
    'BT',  # Bhutan
    'BO',  # Bolivia, Plurinational State of
    'BQ',  # Bonaire, Sint Eustatius and Saba[c]
    'BA',  # Bosnia and Herzegovina
    'BW',  # Botswana
    'BV',  # Bouvet Island
    'BR',  # Brazil
    'IO',  # British Indian Ocean Territory
    'BN',  # Brunei Darussalam
    'BG',  # Bulgaria
    'BF',  # Burkina Faso
    'BI',  # Burundi
    'CV',  # Cabo Verde
    'KH',  # Cambodia
    'CM',  # Cameroon
    'CA',  # Canada
    'KY',  # Cayman Islands
    'CF',  # Central African Republic
    'TD',  # Chad
    'CL',  # Chile
    'CN',  # China[b]
    'CX',  # Christmas Island
    'CC',  # Cocos (Keeling) Islands
    'CO',  # Colombia
    'KM',  # Comoros
    'CG',  # Congo
    'CD',  # Congo, Democratic Republic of the
    'CK',  # Cook Islands
    'CR',  # Costa Rica
    'CI',  # Côte d'Ivoire
    'HR',  # Croatia
    'CU',  # Cuba
    'CW',  # Curaçao
    'CY',  # Cyprus[b]
    'CZ',  # Czechia
    'DK',  # Denmark
    'DJ',  # Djibouti
    'DM',  # Dominica
    'DO',  # Dominican Republic
    'EC',  # Ecuador
    'EG',  # Egypt
    'SV',  # El Salvador
    'GQ',  # Equatorial Guinea
    'ER',  # Eritrea
    'EE',  # Estonia
    'SZ',  # Eswatini
    'ET',  # Ethiopia
    'FK',  # Falkland Islands (Malvinas)[b]
    'FO',  # Faroe Islands
    'FJ',  # Fiji
    'FI',  # Finland
    'FR',  # France
    'GF',  # French Guiana
    'PF',  # French Polynesia
    'TF',  # French Southern Territories
    'GA',  # Gabon
    'GM',  # Gambia
    'GE',  # Georgia
    'DE',  # Germany
    'GH',  # Ghana
    'GI',  # Gibraltar
    'GR',  # Greece
    'GL',  # Greenland
    'GD',  # Grenada
    'GP',  # Guadeloupe
    'GU',  # Guam
    'GT',  # Guatemala
    'GG',  # Guernsey
    'GN',  # Guinea
    'GW',  # Guinea-Bissau
    'GY',  # Guyana
    'HT',  # Haiti
    'HM',  # Heard Island and McDonald Islands
    'VA',  # Holy See
    'HN',  # Honduras
    'HK',  # Hong Kong
    'HU',  # Hungary
    'IS',  # Iceland
    'IN',  # India
    'ID',  # Indonesia
    'IR',  # Iran, Islamic Republic of
    'IQ',  # Iraq
    'IE',  # Ireland
    'IM',  # Isle of Man
    'IL',  # Israel
    'IT',  # Italy
    'JM',  # Jamaica
    'JP',  # Japan
    'JE',  # Jersey
    'JO',  # Jordan
    'KZ',  # Kazakhstan
    'KE',  # Kenya
    'KI',  # Kiribati
    'KP',  # Korea, Democratic People's Republic of
    'KR',  # Korea, Republic of
    'KW',  # Kuwait
    'KG',  # Kyrgyzstan
    'LA',  # Lao People's Democratic Republic
    'LV',  # Latvia
    'LB',  # Lebanon
    'LS',  # Lesotho
    'LR',  # Liberia
    'LY',  # Libya
    'LI',  # Liechtenstein
    'LT',  # Lithuania
    'LU',  # Luxembourg
    'MO',  # Macao
    'MG',  # Madagascar
    'MW',  # Malawi
    'MY',  # Malaysia
    'MV',  # Maldives
    'ML',  # Mali
    'MT',  # Malta
    'MH',  # Marshall Islands
    'MQ',  # Martinique
    'MR',  # Mauritania
    'MU',  # Mauritius
    'YT',  # Mayotte
    'MX',  # Mexico
    'FM',  # Micronesia, Federated States of
    'MD',  # Moldova, Republic of
    'MC',  # Monaco
    'MN',  # Mongolia
    'ME',  # Montenegro
    'MS',  # Montserrat
    'MA',  # Morocco
    'MZ',  # Mozambique
    'MM',  # Myanmar
    'NA',  # Namibia
    'NR',  # Nauru
    'NP',  # Nepal
    'NL',  # Netherlands, Kingdom of the
    'NC',  # New Caledonia
    'NZ',  # New Zealand
    'NI',  # Nicaragua
    'NE',  # Niger
    'NG',  # Nigeria
    'NU',  # Niue
    'NF',  # Norfolk Island
    'MK',  # North Macedonia
    'MP',  # Northern Mariana Islands
    'NO',  # Norway
    'OM',  # Oman
    'PK',  # Pakistan
    'PW',  # Palau
    'PS',  # Palestine, State of[b]
    'PA',  # Panama
    'PG',  # Papua New Guinea
    'PY',  # Paraguay
    'PE',  # Peru
    'PH',  # Philippines
    'PN',  # Pitcairn
    'PL',  # Poland
    'PT',  # Portugal
    'PR',  # Puerto Rico
    'QA',  # Qatar
    'RE',  # Réunion
    'RO',  # Romania
    'RU',  # Russian Federation
    'RW',  # Rwanda
    'BL',  # Saint Barthélemy
    'SH',  # Saint Helena, Ascension and Tristan da Cunha[d]
    'KN',  # Saint Kitts and Nevis
    'LC',  # Saint Lucia
    'MF',  # Saint Martin (French part)
    'PM',  # Saint Pierre and Miquelon
    'VC',  # Saint Vincent and the Grenadines
    'WS',  # Samoa
    'SM',  # San Marino
    'ST',  # Sao Tome and Principe
    'SA',  # Saudi Arabia
    'SN',  # Senegal
    'RS',  # Serbia
    'SC',  # Seychelles
    'SL',  # Sierra Leone
    'SG',  # Singapore
    'SX',  # Sint Maarten (Dutch part)
    'SK',  # Slovakia
    'SI',  # Slovenia
    'SB',  # Solomon Islands
    'SO',  # Somalia
    'ZA',  # South Africa
    'GS',  # South Georgia and the South Sandwich Islands
    'SS',  # South Sudan
    'ES',  # Spain
    'LK',  # Sri Lanka
    'SD',  # Sudan
    'SR',  # Suriname
    'SJ',  # Svalbard and Jan Mayen[e]
    'SE',  # Sweden
    'CH',  # Switzerland
    'SY',  # Syrian Arab Republic
    'TW',  # Taiwan, Province of China[b]
    'TJ',  # Tajikistan
    'TZ',  # Tanzania, United Republic of
    'TH',  # Thailand
    'TL',  # Timor-Leste
    'TG',  # Togo
    'TK',  # Tokelau
    'TO',  # Tonga
    'TT',  # Trinidad and Tobago
    'TN',  # Tunisia
    'TR',  # Türkiye
    'TM',  # Turkmenistan
    'TC',  # Turks and Caicos Islands
    'TV',  # Tuvalu
    'UG',  # Uganda
    'UA',  # Ukraine
    'AE',  # United Arab Emirates
    'GB',  # United Kingdom of Great Britain and Northern Ireland
    'US',  # United States of America
    'UM',  # United States Minor Outlying Islands[f]
    'UY',  # Uruguay
    'UZ',  # Uzbekistan
    'VU',  # Vanuatu
    'VE',  # Venezuela, Bolivarian Republic of
    'VN',  # Viet Nam
    'VG',  # Virgin Islands (British)
    'VI',  # Virgin Islands (U.S.)
    'WF',  # Wallis and Futuna
    'EH',  # Western Sahara[b]
    'YE',  # Yemen
    'ZM',  # Zambia
    'ZW',  # Zimbabwe
}

# See <https://en.wikipedia.org/wiki/Country_code_top-level_domain>.
DNS_COUNTRY_CODES: Final[Set[str]] = (
    ISO3166_1_ALPHA2 - {'BV', 'BL', 'MF', 'SJ', 'GB', 'UM'}
) | {'AC', 'EN', 'EU', 'UK'}

# Approximation.
ENGLISH_DNS_COUNTRY_CODES: Final[Set[str]] = {'AU', 'CA', 'EN', 'UK', 'US'}
NON_ENGLISH_DNS_COUNTRY_CODES: Final[Set[str]] = (
    DNS_COUNTRY_CODES - ENGLISH_DNS_COUNTRY_CODES
)

# Language tags.
LANGUAGE_TAGS: Final[Set[str]] = {
    'aa',                       # Afar
    'ab',                       # Abkhazian
    'ae',                       # Avestan
    'af',                       # Afrikaans
    'ak',                       # Akan
    'am',                       # Amharic
    'an',                       # Aragonese
    'ar',                       # Arabic
    'as',                       # Assamese
    'av',                       # Avaric
    'ay',                       # Aymara
    'az',                       # Azerbaijani
    'ba',                       # Bashkir
    'be',                       # Belarusian
    'bg',                       # Bulgarian
    'bh',                       # Bihari
    'bi',                       # Bislama
    'bm',                       # Bambara
    'bn',                       # Bengali (Bangla)
    'bo',                       # Tibetan
    'br',                       # Breton
    'bs',                       # Bosnian
    'ca',                       # Catalan
    'ce',                       # Chechen
    'ch',                       # Chamorro
    'co',                       # Corsican
    'cr',                       # Cree
    'cs',                       # Czech
    'cu',                       # Old Church Slavonic, Old Bulgarian
    'cv',                       # Chuvash
    'cy',                       # Welsh
    'da',                       # Danish
    'de',                       # German
    'dv',                       # Divehi, Dhivehi, Maldivian
    'dz',                       # Dzongkha
    'ee',                       # Ewe
    'el',                       # Greek
    'en',                       # English
    'eo',                       # Esperanto
    'es',                       # Spanish
    'et',                       # Estonian
    'eu',                       # Basque
    'fa',                       # Persian (Farsi)
    'ff',                       # Fula, Fulah, Pulaar, Pular
    'fi',                       # Finnish
    'fj',                       # Fijian
    'fo',                       # Faroese
    'fr',                       # French
    'fy',                       # Western Frisian
    'ga',                       # Irish
    'gd',                       # Gaelic (Scottish)
    'gl',                       # Galician
    'gn',                       # Guarani
    'gu',                       # Gujarati
    'gv',                       # Gaelic (Manx)
    'gv',                       # Manx
    'ha',                       # Hausa
    'he',                       # Hebrew
    'hi',                       # Hindi
    'ho',                       # Hiri Motu
    'hr',                       # Croatian
    'ht',                       # Haitian Creole
    'hu',                       # Hungarian
    'hy',                       # Armenian
    'hz',                       # Herero
    'ia',                       # Interlingua
    'id',                       # Indonesian
    'ie',                       # Interlingue
    'ig',                       # Igbo
    'ii',                       # Nuosu
    'ii',                       # Sichuan Yi
    'ik',                       # Inupiak
    'in',                       # Indonesian
    'io',                       # Ido
    'is',                       # Icelandic
    'it',                       # Italian
    'iu',                       # Inuktitut
    'ja',                       # Japanese
    'ji',                       # Yiddish
    'jv',                       # Javanese
    'ka',                       # Georgian
    'kg',                       # Kongo
    'ki',                       # Kikuyu
    'kj',                       # Kwanyama
    'kk',                       # Kazakh
    'kl',                       # Greenlandic
    'kl',                       # Kalaallisut, Greenlandic
    'km',                       # Khmer
    'kn',                       # Kannada
    'ko',                       # Korean
    'kr',                       # Kanuri
    'ks',                       # Kashmiri
    'ku',                       # Kurdish
    'kv',                       # Komi
    'kw',                       # Cornish
    'ky',                       # Kyrgyz
    'la',                       # Latin
    'lb',                       # Luxembourgish
    'lg',                       # Luganda, Ganda
    'li',                       # Limburgish ( Limburger)
    'ln',                       # Lingala
    'lo',                       # Lao
    'lt',                       # Lithuanian
    'lu',                       # Luga-Katanga
    'lv',                       # Latvian (Lettish)
    'mg',                       # Malagasy
    'mh',                       # Marshallese
    'mi',                       # Maori
    'mk',                       # Macedonian
    'ml',                       # Malayalam
    'mn',                       # Mongolian
    'mo',                       # Moldavian
    'mr',                       # Marathi
    'ms',                       # Malay
    'mt',                       # Maltese
    'my',                       # Burmese
    'na',                       # Nauru
    'nb',                       # Norwegian bokmål
    'nd',                       # Northern Ndebele
    'ne',                       # Nepali
    'ng',                       # Ndonga
    'nl',                       # Dutch
    'nn',                       # Norwegian nynorsk
    'no',                       # Norwegian
    'nr',                       # Southern Ndebele
    'nv',                       # Navajo
    'ny',                       # Chichewa, Chewa, Nyanja
    'oc',                       # Occitan
    'oj',                       # Ojibwe
    'om',                       # Oromo (Afaan Oromo)
    'or',                       # Oriya
    'os',                       # Ossetian
    'pa',                       # Punjabi (Eastern)
    'pi',                       # Pāli
    'pl',                       # Polish
    'ps',                       # Pashto, Pushto
    'pt',                       # Portuguese
    'qu',                       # Quechua
    'rm',                       # Romansh
    'rn',                       # Kirundi
    'ro',                       # Romanian
    'ru',                       # Russian
    'rw',                       # Kinyarwanda (Rwanda)
    'sa',                       # Sanskrit
    'sd',                       # Sindhi
    'se',                       # Sami
    'sg',                       # Sango
    'sh',                       # Serbo-Croatian
    'si',                       # Sinhalese
    'sk',                       # Slovak
    'sl',                       # Slovenian
    'sm',                       # Samoan
    'sn',                       # Shona
    'so',                       # Somali
    'sq',                       # Albanian
    'sr',                       # Serbian
    'ss',                       # Siswati
    'ss',                       # Swati
    'st',                       # Sesotho
    'su',                       # Sundanese
    'sv',                       # Swedish
    'sw',                       # Swahili (Kiswahili)
    'ta',                       # Tamil
    'te',                       # Telugu
    'tg',                       # Tajik
    'th',                       # Thai
    'ti',                       # Tigrinya
    'tk',                       # Turkmen
    'tl',                       # Tagalog
    'tn',                       # Setswana
    'to',                       # Tonga
    'tr',                       # Turkish
    'ts',                       # Tsonga
    'tt',                       # Tatar
    'tw',                       # Twi
    'ty',                       # Tahitian
    'ug',                       # Uyghur
    'uk',                       # Ukrainian
    'ur',                       # Urdu
    'uz',                       # Uzbek
    've',                       # Venda
    'vi',                       # Vietnamese
    'vo',                       # Volapük
    'wa',                       # Wallon
    'wo',                       # Wolof
    'xh',                       # Xhosa
    'yi',                       # Yiddish
    'yo',                       # Yoruba
    'za',                       # Zhuang, Chuang
    'zh',                       # Chinese
    'zh-Hans',                  # Chinese (Simplified)
    'zh-Hant',                  # Chinese (Traditional)
    'zu',                       # Zulu
}

ENGLISH_LANGUAGE_TAGS: Final[Set[str]] = {'en'}

NON_ENGLISH_LAGUAGE_TAGS: Final[Set[str]] =\
    LANGUAGE_TAGS - ENGLISH_LANGUAGE_TAGS


def get_ignored_url_patterns(
        exclude: Collection[str] = ()
) -> Sequence[str]:
    ignored = set(map(
        lambda s: s.upper(),
        (NON_ENGLISH_DNS_COUNTRY_CODES | NON_ENGLISH_LAGUAGE_TAGS)))\
        - set(map(lambda s: s.upper(), exclude))
    ne = '(' + '|'.join(sorted(map(
        lambda s: f'{s.upper()}|{s.lower()}', ignored))) + ')'
    return [
        fr'^https?://[^/]*\.{ne}/',
        fr'^https?://{ne}\.[^/]*/',
        fr'^https?://en\.vikidia\.org/wiki/{ne}:',
        r'^https?://www\.enciclopedia\.cat/',
        fr'^https?://{ne}\.cathopedia\.org/',
        fr'^https?://{ne}\.quora\.com/',
        fr'^https?://{ne}\.vikidia\.org/',
        (
            r'^https://wikidata-externalid-url\.toolforge\.org/'
            fr'?p=[0-9]*&url_prefix=https?://[^/]*\.{ne}/'
        ),
    ]
