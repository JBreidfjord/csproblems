from dataclasses import dataclass


@dataclass
class Item:
    name: str
    weight: int
    value: int


item_list = [
    Item("Axe", 32_252, 68_674),
    Item("Bronze coin", 225_790, 471_010),
    Item("Crown", 468_164, 944_620),
    Item("Diamond statue", 489_494, 962_094),
    Item("Emerald belt", 35_384, 78_344),
    Item("Fossil", 265_590, 579_152),
    Item("Gold coin", 497_911, 902_698),
    Item("Helmet", 800_493, 1_686_515),
    Item("Ink", 823_576, 1_688_691),
    Item("Jewel box", 552_202, 1_056_157),
    Item("Knife", 323_618, 677_562),
    Item("Long sword", 382_846, 833_132),
    Item("Mask", 44_676, 99_192),
    Item("Necklace", 169_738, 376_418),
    Item("Opal badge", 610_876, 1_253_986),
    Item("Pearls", 854_190, 1_853_562),
    Item("Quiver", 671_123, 1_320_297),
    Item("Ruby ring", 698_180, 1_301_637),
    Item("Silver bracelet", 446_517, 859_835),
    Item("Timepiece", 909_620, 1_677_534),
    Item("Uniform", 904_818, 1_910_501),
    Item("Venom potion", 730_061, 1_528_646),
    Item("Wool scarf", 931_932, 1_827_477),
    Item("Crossbow", 952_360, 2_068_204),
    Item("Yesteryear book", 926_023, 1_746_556),
    Item("Zinc cup", 978_724, 2_100_851),
]

