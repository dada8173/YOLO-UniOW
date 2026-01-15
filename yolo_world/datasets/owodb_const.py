import itertools

__all__ = ["VOC_COCO_CLASS_NAMES", "VOC_CLASS_NAMES_COCOFIED", "BASE_VOC_CLASS_NAMES", "UNK_CLASS"]


#OWOD splits
VOC_CLASS_NAMES_COCOFIED = [
    "airplane",  "dining table", "motorcycle",
    "potted plant", "couch", "tv"
]

BASE_VOC_CLASS_NAMES = [
    "aeroplane", "diningtable", "motorbike",
    "pottedplant",  "sofa", "tvmonitor"
]
UNK_CLASS = ["unknown"]

VOC_COCO_CLASS_NAMES={}


T1_CLASS_NAMES = [
    "aeroplane","bicycle","bird","boat","bus","car",
    "cat","cow","dog","horse","motorbike","sheep","train",
    "elephant","bear","zebra","giraffe","truck","person"
]

T2_CLASS_NAMES = [
    "traffic light","fire hydrant","stop sign",
    "parking meter","bench","chair","diningtable",
    "pottedplant","backpack","umbrella","handbag",
    "tie","suitcase","microwave","oven","toaster","sink",
    "refrigerator","bed","toilet","sofa"
]

T3_CLASS_NAMES = [
    "frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard",
    "surfboard","tennis racket","banana","apple","sandwich",
    "orange","broccoli","carrot","hot dog","pizza","donut","cake"
]

T4_CLASS_NAMES = [
    "laptop","mouse","remote","keyboard","cell phone","book",
    "clock","vase","scissors","teddy bear","hair drier","toothbrush",
    "wine glass","cup","fork","knife","spoon","bowl","tvmonitor","bottle"
]

VOC_COCO_CLASS_NAMES["SOWODB"] = tuple(itertools.chain(T1_CLASS_NAMES, T2_CLASS_NAMES, T3_CLASS_NAMES, T4_CLASS_NAMES, UNK_CLASS))


VOC_CLASS_NAMES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

T2_CLASS_NAMES = [
    "truck", "traffic light", "fire hydrant", "stop sign", "parking meter",
    "bench", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    "microwave", "oven", "toaster", "sink", "refrigerator"
]

T3_CLASS_NAMES = [
    "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake"
]

T4_CLASS_NAMES = [
    "bed", "toilet", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl"
]
VOC_COCO_CLASS_NAMES["MOWODB"] = tuple(itertools.chain(VOC_CLASS_NAMES, T2_CLASS_NAMES, T3_CLASS_NAMES, T4_CLASS_NAMES, UNK_CLASS))

T1_CLASS_NAMES = [
        'vehicle.bicycle',
        'vehicle.motorcycle',
        'vehicle.car',
        'vehicle.bus.bendy',
        'vehicle.bus.rigid',
        'vehicle.truck',
        'vehicle.emergency.ambulance',
        'vehicle.emergency.police',
        'vehicle.construction',
        'vehicle.trailer'
]

T2_CLASS_NAMES = [
        'human.pedestrian.adult',
        'human.pedestrian.child',
        'human.pedestrian.wheelchair',
        'human.pedestrian.stroller',
        'human.pedestrian.personal_mobility',
        'human.pedestrian.police_officer',
        'human.pedestrian.construction_worker'
]

T3_CLASS_NAMES = [
        'movable_object.barrier',
        'movable_object.trafficcone',
        'movable_object.pushable_pullable',
        'movable_object.debris',
        'static_object.bicycle_rack',
        'animal'
]

VOC_COCO_CLASS_NAMES["nuOWODB"] = tuple(itertools.chain(T1_CLASS_NAMES, T2_CLASS_NAMES, T3_CLASS_NAMES, UNK_CLASS))

# GroceryOWOD
GROCERY_T1_CLASS_NAMES = ['category_3', 'category_10', 'category_9']
GROCERY_T2_CLASS_NAMES = ['category_5', 'category_6', 'category_8']
GROCERY_T3_CLASS_NAMES = ['category_1', 'category_7', 'category_4']
GROCERY_T4_CLASS_NAMES = ['category_2']
VOC_COCO_CLASS_NAMES["GroceryOWOD"] = tuple(itertools.chain(
    GROCERY_T1_CLASS_NAMES, 
    GROCERY_T2_CLASS_NAMES, 
    GROCERY_T3_CLASS_NAMES, 
    GROCERY_T4_CLASS_NAMES, 
    UNK_CLASS))

# GroZi-120 OWOD
GROZI_T1_CLASS_NAMES = [
    "Bausch & Lomb Renu All in One Multi Purpose Solution", "Chex Mix", "Gardetto's Original Recipe",
    "Honey Nut Cheerios (General Mills)", "Wrigleys Extra peppermint gum", "doublemint gum",
    "big red gum", "Vicks DayQuil LiquiCaps", "Cheez-It",
    "Hershey Milk Chocolate with Almonds", "Twix Cookie Bar", "Snickers",
    "CLOROX ULTRA LIQUID REG", "CLOROX 2 LIQUID 44 OZ", "Gatorade Lemon-Lime",
    "SNYDER PRETZEL OLD TYME", "POCKETCOMB", "Motrin IB Ibuprofen Tablets USP",
    "Neosporin Original", "El Sabroso Salsitas Salsa Chips", "Tic Tac Wintergreen",
    "DENTYNE ICE ARTIC CHILL", "Certs Peppermint", "Genuine BAYER Aspirin, tablets (325 mg)",
    "Nissin Ramen Noodles - Chicken", "26OZ BLUE WINDEX", "Morton Salt, Iodized",
    "Ziploc Sandwich Bags", "Aleve Caplets", "Kit Kat King Size"
]

GROZI_T2_CLASS_NAMES = [
    "Reese's Pieces", "Kleenex Tissue", "Chap Stick Lip Balm",
    "REGULAR 33OZ TIDE POWDER", "Pringles Sour Cream & Onion", "Pringles Pizza-licious",
    "Starburst Original Fruit", "Skittles (Original)", "Claritin Allergy",
    "French's Classic Yellow Mustard", "FORMULA 409", "Monster Energy Beverage",
    "Mentos (Mint)", "Tapatio - Salsa Picante", "Dove Anti-Perspirant/Deodorant Fresh Invisible Solid",
    "Band-Aid flexible fabric", "CARMEX EZ-ON APPLICATOR", "diet ROCKSTAR Energy Drink",
    "VICKS NYQUIL COUGH CHERRY 6 OZ", "Nestle Crunch", "Lay's Classic",
    "ARM + HAMMER BAKING SODA", "MARTINELLI APPLE JUICE", "Diet Coke with Lime",
    "Sprite 12oz Can", "Propel Fitness Water", "AVIATOR POKER CARDS",
    "Act II Butter Lover's Popcorn", "Dr Pepper", "Tylenol PM Gelcaps Extra Strength"
]

GROZI_T3_CLASS_NAMES = [
    "Red Bull Sugarfree", "Blistex \"Silk & Shine\" Lip gloss and sunscreen SPF 15", "NYQUIL LQCAPS PSE FREE 12 CT",
    "LINDOR CANDY", "Lindt Excellence 70% Cocoa Dark Chocolate", "Haribo Gold-Bears Gummi Candy",
    "Ritter Sport White Chocolate with Whole Hazelnuts", "TOBLERONE MILK CHOCOLATE", "Manner Milk Chocolate Cream Filled Wafers",
    "PEPPERIDGE FARM MILK CHOCOLATE MACADAMIA COOKIES", "Pepperidge Farm Milano Cookies, Double Chocolate", "Tylenol  Extra Strength Caplets",
    "Frappuccino Coffee", "Yoo-Hoo Chocolate Drink", "Bull's-Eye BBQ Sauce Original",
    "Symphony with almonds and toffee chips", "Nabisco Nilla Wafers", "BEEF JERKY",
    "David Sunflower Seeds", "Hunt's Tomato Sauce", "kotex lightdays",
    "Always thin pantiliners", "Liption Tea", "Soft Scrub with Bleach Cleanser",
    "RAID Flying Insect Killer", "SHOUT LAUNDRY STAIN REMOVER", "Gillette Foamy Shaving Cream, Regular",
    "Campbell's Tomato Soup - Microwavable bowl", "Vivarin", "Chef Boyardee Beef Ravioli"
]

GROZI_T4_CLASS_NAMES = [
    "LIFESTYLES ULTRA SENSITIVE", "Crystal Geyser Water", "Luna Bar nutz chocolate",
    "KELL RAISIN BRAN 15 OZ 121459", "GM HNY NUT CHEERIOS CEREAL CUP", "Pepto-Bismol",
    "PEPTO BISMOL CHERRY TAB 30 CT", "Tabasco Brand Pepper Sauce", "alka Seltzer Plus cold",
    "Jack Links: Extreme Snack Stick", "TROJAN ULTRA FIT CONDOMS", "Trojan-Enz lubricated condoms",
    "colgate plus toothbrush", "Nabisco Wheat Thins Crackers Original", "Mountain Dew, Single Bottle",
    "GUMMI FROGS", "Mini Chips Ahoy!", "Jif Creamy Peanut Butter",
    "Glade Potpourri Spray ,Powder Fresh ,9oz.", "Campbells Cream of Chicken soup", "CAMPBELL CHUNKY CLASSIC CHICKEN NOODLE",
    "A-1 STEAK SAUCE", "Rockstar Energy Drink", "KODAK COLOR SLIDE 100 EB135-36",
    "Tostitos Scoops", "RITTER CAPPUCCINO 100G", "Sour Cream & Onion Ruffles",
    "Ben & Jerry's Ice Cream World's Best Vanilla", "Nabisco Flavor Crisps/Flavor Originals Snack Crackers Baked Vegetable Thins", "Yoplait Yogurt Original Strawberry Banana 99% Fat Free S70"
]

VOC_COCO_CLASS_NAMES['GroZi120OWOD'] = tuple(itertools.chain(
    GROZI_T1_CLASS_NAMES,
    GROZI_T2_CLASS_NAMES,
    GROZI_T3_CLASS_NAMES,
    GROZI_T4_CLASS_NAMES,
    UNK_CLASS
))
