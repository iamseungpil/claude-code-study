# -*- coding: utf-8 -*-
"""
everything.py - Legacy Code Refactoring Challenge

이 파일은 누군가가 "나중에 정리하지 뭐" 하고 만든 파일입니다.
두 개의 완전히 다른 도메인의 코드가 섞여 있습니다.

TODO: 나중에 파일 분리하기
TODO: 테스트 더 추가하기
FIXME: 이거 왜 이렇게 복잡하지?

당신의 미션:
1. 이 파일을 분석하여 서로 다른 두 프로젝트를 식별하세요
2. 각 프로젝트를 별도의 폴더/모듈로 분리하세요
3. 코드를 리팩토링하여 가독성을 개선하세요
4. 모든 테스트가 통과하는지 확인하세요
5. README.md를 작성하세요
"""

import unittest


# ================================================================
# 뭔가 아이템 관련된 것 같은데...
# ================================================================

class Item:
    def __init__(self, name, sell_in, quality):
        self.name = name
        self.sell_in = sell_in
        self.quality = quality

    def __repr__(self):
        return "%s, %s, %s" % (self.name, self.sell_in, self.quality)


class GildedRose(object):
    """상점 재고 관리 시스템인 것 같음"""

    def __init__(self, items):
        self.items = items

    def update_quality(self):
        for item in self.items:
            if item.name != "Aged Brie" and item.name != "Backstage passes to a TAFKAL80ETC concert":
                if item.quality > 0:
                    if item.name != "Sulfuras, Hand of Ragnaros":
                        item.quality = item.quality - 1
            else:
                if item.quality < 50:
                    item.quality = item.quality + 1
                    if item.name == "Backstage passes to a TAFKAL80ETC concert":
                        if item.sell_in < 11:
                            if item.quality < 50:
                                item.quality = item.quality + 1
                        if item.sell_in < 6:
                            if item.quality < 50:
                                item.quality = item.quality + 1
            if item.name != "Sulfuras, Hand of Ragnaros":
                item.sell_in = item.sell_in - 1
            if item.sell_in < 0:
                if item.name != "Aged Brie":
                    if item.name != "Backstage passes to a TAFKAL80ETC concert":
                        if item.quality > 0:
                            if item.name != "Sulfuras, Hand of Ragnaros":
                                item.quality = item.quality - 1
                    else:
                        item.quality = item.quality - item.quality
                else:
                    if item.quality < 50:
                        item.quality = item.quality + 1


# ================================================================
# 이건 또 뭐지? 게임 점수 같은데...
# ================================================================

class TennisGame:
    """테니스 게임 점수 계산기"""

    def __init__(self, player1_name, player2_name):
        self.player1_name = player1_name
        self.player2_name = player2_name
        self.p1points = 0
        self.p2points = 0

    def won_point(self, player_name):
        if player_name == self.player1_name:
            self.p1points += 1
        else:
            self.p2points += 1

    def score(self):
        result = ""
        temp_score = 0
        if self.p1points == self.p2points:
            result = {
                0: "Love-All",
                1: "Fifteen-All",
                2: "Thirty-All",
            }.get(self.p1points, "Deuce")
        elif self.p1points >= 4 or self.p2points >= 4:
            minus_result = self.p1points - self.p2points
            if minus_result == 1:
                result = "Advantage " + self.player1_name
            elif minus_result == -1:
                result = "Advantage " + self.player2_name
            elif minus_result >= 2:
                result = "Win for " + self.player1_name
            else:
                result = "Win for " + self.player2_name
        else:
            for i in range(1, 3):
                if i == 1:
                    temp_score = self.p1points
                else:
                    result += "-"
                    temp_score = self.p2points
                result += {
                    0: "Love",
                    1: "Fifteen",
                    2: "Thirty",
                    3: "Forty",
                }[temp_score]
        return result


# ================================================================
# 테스트 코드들 (일부만 있음, 나머지는 어디 갔지?)
# ================================================================

class GildedRoseTest(unittest.TestCase):
    """Gilded Rose 테스트"""

    def test_normal_item_decreases_quality(self):
        items = [Item("Normal Item", 10, 20)]
        gilded_rose = GildedRose(items)
        gilded_rose.update_quality()
        self.assertEqual(9, items[0].sell_in)
        self.assertEqual(19, items[0].quality)

    def test_quality_never_negative(self):
        items = [Item("Normal Item", 10, 0)]
        gilded_rose = GildedRose(items)
        gilded_rose.update_quality()
        self.assertEqual(0, items[0].quality)

    def test_aged_brie_increases_quality(self):
        items = [Item("Aged Brie", 10, 20)]
        gilded_rose = GildedRose(items)
        gilded_rose.update_quality()
        self.assertEqual(21, items[0].quality)

    def test_quality_never_exceeds_50(self):
        items = [Item("Aged Brie", 10, 50)]
        gilded_rose = GildedRose(items)
        gilded_rose.update_quality()
        self.assertEqual(50, items[0].quality)

    def test_sulfuras_never_changes(self):
        items = [Item("Sulfuras, Hand of Ragnaros", 10, 80)]
        gilded_rose = GildedRose(items)
        gilded_rose.update_quality()
        self.assertEqual(10, items[0].sell_in)
        self.assertEqual(80, items[0].quality)

    def test_backstage_pass_increases_quality(self):
        items = [Item("Backstage passes to a TAFKAL80ETC concert", 15, 20)]
        gilded_rose = GildedRose(items)
        gilded_rose.update_quality()
        self.assertEqual(21, items[0].quality)

    def test_backstage_pass_increases_by_2_when_10_days_or_less(self):
        items = [Item("Backstage passes to a TAFKAL80ETC concert", 10, 20)]
        gilded_rose = GildedRose(items)
        gilded_rose.update_quality()
        self.assertEqual(22, items[0].quality)

    def test_backstage_pass_increases_by_3_when_5_days_or_less(self):
        items = [Item("Backstage passes to a TAFKAL80ETC concert", 5, 20)]
        gilded_rose = GildedRose(items)
        gilded_rose.update_quality()
        self.assertEqual(23, items[0].quality)

    def test_backstage_pass_quality_drops_to_0_after_concert(self):
        items = [Item("Backstage passes to a TAFKAL80ETC concert", 0, 20)]
        gilded_rose = GildedRose(items)
        gilded_rose.update_quality()
        self.assertEqual(0, items[0].quality)

    def test_expired_item_degrades_twice_as_fast(self):
        items = [Item("Normal Item", 0, 10)]
        gilded_rose = GildedRose(items)
        gilded_rose.update_quality()
        self.assertEqual(8, items[0].quality)


class TennisGameTest(unittest.TestCase):
    """Tennis Game 테스트"""

    def play_game(self, p1_points, p2_points, p1_name="player1", p2_name="player2"):
        game = TennisGame(p1_name, p2_name)
        for i in range(max(p1_points, p2_points)):
            if i < p1_points:
                game.won_point(p1_name)
            if i < p2_points:
                game.won_point(p2_name)
        return game

    def test_love_all(self):
        game = self.play_game(0, 0)
        self.assertEqual("Love-All", game.score())

    def test_fifteen_all(self):
        game = self.play_game(1, 1)
        self.assertEqual("Fifteen-All", game.score())

    def test_thirty_all(self):
        game = self.play_game(2, 2)
        self.assertEqual("Thirty-All", game.score())

    def test_deuce(self):
        game = self.play_game(3, 3)
        self.assertEqual("Deuce", game.score())

    def test_fifteen_love(self):
        game = self.play_game(1, 0)
        self.assertEqual("Fifteen-Love", game.score())

    def test_love_fifteen(self):
        game = self.play_game(0, 1)
        self.assertEqual("Love-Fifteen", game.score())

    def test_forty_love(self):
        game = self.play_game(3, 0)
        self.assertEqual("Forty-Love", game.score())

    def test_win_for_player1(self):
        game = self.play_game(4, 0)
        self.assertEqual("Win for player1", game.score())

    def test_win_for_player2(self):
        game = self.play_game(0, 4)
        self.assertEqual("Win for player2", game.score())

    def test_advantage_player1(self):
        game = self.play_game(4, 3)
        self.assertEqual("Advantage player1", game.score())

    def test_advantage_player2(self):
        game = self.play_game(3, 4)
        self.assertEqual("Advantage player2", game.score())

    def test_win_after_deuce(self):
        game = self.play_game(6, 4)
        self.assertEqual("Win for player1", game.score())


# ================================================================
# 메인 실행부 (이것도 정리가 필요함)
# ================================================================

def run_gilded_rose_demo():
    """Gilded Rose 데모"""
    print("=" * 50)
    print("Gilded Rose Demo")
    print("=" * 50)

    items = [
        Item("+5 Dexterity Vest", 10, 20),
        Item("Aged Brie", 2, 0),
        Item("Elixir of the Mongoose", 5, 7),
        Item("Sulfuras, Hand of Ragnaros", 0, 80),
        Item("Backstage passes to a TAFKAL80ETC concert", 15, 20),
    ]

    gilded_rose = GildedRose(items)

    for day in range(5):
        print(f"\n---- Day {day} ----")
        for item in items:
            print(item)
        gilded_rose.update_quality()


def run_tennis_demo():
    """Tennis Game 데모"""
    print("\n" + "=" * 50)
    print("Tennis Game Demo")
    print("=" * 50)

    game = TennisGame("Federer", "Nadal")

    # 시뮬레이션
    points = ["Federer", "Nadal", "Federer", "Federer", "Nadal", "Nadal", "Federer"]

    for player in points:
        game.won_point(player)
        print(f"{player} scores! Current: {game.score()}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # 테스트 실행
        unittest.main(argv=[''], exit=False)
    else:
        # 데모 실행
        run_gilded_rose_demo()
        run_tennis_demo()

        print("\n" + "=" * 50)
        print("테스트 실행: python everything.py test")
        print("=" * 50)
