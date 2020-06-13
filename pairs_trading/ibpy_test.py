from ib.opt import Connection, message
from ib.ext.Contract import Contract
from ib.ext.Order import Order


def error_handler(msg):
    """Handles the capturing of error messages"""
    print("Server Error: %s" % msg)


def reply_handler(msg):
    """Handles of server replies"""
    print("Server Response: %s, %s" % (msg.typeName, msg))


def account_handler(msg):
    print("Account message: ", msg)


def tick_handler(msg):
    print("Tick message:", msg)


def create_contract(symbol, sec_type, exch, prim_exch, curr):
    """Create a Contract object defining what will
    be purchased, at which exchange and in which currency.

    symbol - The ticker symbol for the contract
    sec_type - The security type for the contract ('STK' is 'stock')
    exch - The exchange to carry out the contract on
    prim_exch - The primary exchange to carry out the contract on
    curr - The currency in which to purchase the contract"""
    contract = Contract()
    contract.m_symbol = symbol
    contract.m_secType = sec_type
    contract.m_exchange = exch
    contract.m_primaryExch = prim_exch
    contract.m_currency = curr
    return contract


def create_order(order_type, quantity, action):
    """Create an Order object (Market/Limit) to go long/short.

    order_type - 'MKT', 'LMT' for Market or Limit orders
    quantity - Integral number of assets to order
    action - 'BUY' or 'SELL'"""
    order = Order()
    order.m_orderType = order_type
    order.m_totalQuantity = quantity
    order.m_action = action
    return order


def main():
    # Connect to the Trader Workstation (TWS) running on the
    # usual port of 7496, with a clientId of 100
    # (The clientId is chosen by us and we will need
    # separate IDs for both the execution connection and
    # market data connection)
    tws_conn = Connection.create(port=7496, clientId=100)


    # Assign the error handling function defined above
    # to the TWS connection
    # tws_conn.register(error_handler, 'Error')

    # Assign all of the server reply messages to the
    # reply_handler function defined above
    tws_conn.registerAll(reply_handler)

    tws_conn.register(account_handler, 'UpdateAccountValue')
    tws_conn.register(tick_handler, message.tickSize, message.tickPrice)

    tws_conn.connect()

    # Create an order ID which is 'global' for this session. This
    # will need incrementing once new orders are submitted.
    order_id = 4

    # Create a contract in GOOG stock via SMART order routing
    goog_contract = create_contract('GOOG', 'STK', 'SMART', 'SMART', 'USD')

    # Go long 100 shares of Google
    goog_order = create_order('MOC', 100, 'BUY')

    # Use the connection to the send the order to IB
    print("Placing order")
    tws_conn.placeOrder(order_id, goog_contract, goog_order)
    print("Finished placing order")

    # Disconnect from TWS
    tws_conn.disconnect()


if __name__ == '__main__':
    main()
