import contextlib
import io

from conftest import import_from_project


def test_buy_lots_of_fruit_totals_known_order():
    fruit = import_from_project("project0", "buyLotsOfFruit")

    order = [("apples", 2.0), ("pears", 3.0), ("limes", 4.0)]

    assert fruit.buyLotsOfFruit(order) == 12.25


def test_buy_lots_of_fruit_rejects_unknown_item():
    fruit = import_from_project("project0", "buyLotsOfFruit")

    with contextlib.redirect_stdout(io.StringIO()) as output:
        result = fruit.buyLotsOfFruit([("dragonfruit", 1.0)])

    assert result is None
    assert "Fruit not in Order" in output.getvalue()


def test_shop_smart_selects_lowest_total_shop():
    shop = import_from_project("project0", "shop")
    shop_smart = import_from_project("project0", "shopSmart")

    with contextlib.redirect_stdout(io.StringIO()):
        shop1 = shop.FruitShop("shop1", {"apples": 2.0, "oranges": 1.0})
        shop2 = shop.FruitShop("shop2", {"apples": 1.0, "oranges": 5.0})

    assert shop_smart.shopSmart([("apples", 1.0), ("oranges", 3.0)], [shop1, shop2]).getName() == "shop1"
    assert shop_smart.shopSmart([("apples", 3.0)], [shop1, shop2]).getName() == "shop2"


def test_fruit_shop_unknown_item_returns_none():
    shop = import_from_project("project0", "shop")

    with contextlib.redirect_stdout(io.StringIO()) as output:
        fruit_shop = shop.FruitShop("corner", {"apples": 2.0})
        result = fruit_shop.getCostPerPound("pears")

    assert result is None
    assert "Sorry we don't have pears" in output.getvalue()
