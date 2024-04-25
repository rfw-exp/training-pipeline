from typing import Optional

import pandera as pa
from pandera.typing import Index, Series


class ConsumerSchema(pa.DataFrameModel):
    id: Index[str]
    number_of_open_accounts: Series[int]
    total_credit_limit: Series[int]
    total_balance: Series[float]
    number_of_accounts_in_arrears: Series[int]
    fraud_flag: Series[Optional[bool]]
