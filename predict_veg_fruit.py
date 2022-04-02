import tensorflow as tf
import numpy as np
from keras.preprocessing import image


def predict_image(userimage, model):
    category={
        0: 'Apple', 1: 'Avocado', 2: 'Banana', 3: 'Bean', 4: 'Bitter Gourd', 5: 'Blueberry', 
        6: 'Bottle Gourd', 7: 'Brinjal', 8: 'Broccoli', 9: 'Cabbage', 10: 'Capsicum', 11: 'Carrot', 
        12: 'Cauliflower', 13: 'Coconut', 14: 'Cucumber', 15: 'Dragonfruit', 16: 'Durian', 17: 'Grape', 
        18: 'Guava', 19: 'Kiwi', 20: 'Lemon', 21: 'Lime', 22: 'Longan', 23: 'Mango', 
        24: 'Muskmelon', 25: 'Orange', 26: 'Papaya', 27: 'Passion Fruit', 28: 'Pear', 29: 'Pineapple', 
        30: 'Plumcot', 31: 'Pomegranate', 32: 'Pomelo', 33: 'Potato', 34: 'Pumpkin', 35: 'Radish', 
        36: 'Strawberry', 37: 'Tomato', 38: 'Watermelon',
    }


    thumbnail={
        0:  '/images/Apple.jpg',
        1:  '/images/Avocado.jpg',
        2:  '/images/Banana.jpg',
        3:  '/images/Bean.jpg',
        4:  '/images/Bittergourd.jpg',
        5:  '/images/Blueberry.jpg',
        6:  '/images/Bottlegourd.jpg',
        7:  '/images/Brinjal.jpg',
        8:  '/images/Broccoli.jpg',
        9:  '/images/Cabbage.jpg',
        10: '/images/Capsicum.jpg',
        11: '/images/Carrot.jpg',
        12: '/images/Cauliflower.jpg',
        13: '/images/Coconut.jpg',
        14: '/images/Cucumber.jpg',
        15: '/images/Dragonfruit.jpg',
        16: '/images/Durian.jpg',
        17: '/images/Grape.jpg',
        18: '/images/Guava.jpg',
        19: '/images/Kiwi.jpg',
        20: '/images/Lemon.jpg',
        21: '/images/Lime.jpg',
        22: '/images/Longan.jpg',
        23: '/images/Mango.jpg',
        24: '/images/Muskmelon.jpg',
        25: '/images/Orange.jpg',
        26: '/images/Papaya.jpg',
        27: '/images/Passionfruit.jpg',
        28: '/images/Pear.jpg',
        29: '/images/Pineapple.jpg',
        30: '/images/Plumcot.jpg',
        31: '/images/Pomegranate.jpg',
        32: '/images/Pomelo.jpg',
        33: '/images/Potato.jpg',
        34: '/images/Pumpkin.jpg',
        35: '/images/Radish.jpg',
        36: '/images/Strawberry.jpg',
        37: '/images/Tomato.jpg',
        38: '/images/Watermelon.jpg',
    }
    writeup={
        0:  "Did you know apples are among the world’s most popular fruits? They are high in fiber, vitamin C, various antioxidants and low in calories. As such, an Apple a day may really keep the doctor away.",
        1:  "Did you know avocado contains mostly healthy monounsaturated fat? This is helpful in lowering your LDL or 'bad' cholesterol and reducing the risk of heart attacks. Time to order more Avocado Toast!",
        2:  "Did you know bananas are a great potassium-loaded food which also contains carbohydrates, vitamin C, and magnesium? Eating it helps in maintaining our healthy blood pressure. This is pretty a-peel-ing!",
        3:  "Did you know beans are low in calories while providing filling fiber and protein? They are high in various vitamins and minerals too. Do not pick out your beans in your fried rice next time!",
        4:  "Did you know bitter gourd is low in calories and carbs and high in beneficial fiber? They also enhances body immunity. Another reason to eat the Bitter Gourd with Scrambled Eggs that your Mum cooked.",
        5:  "Did you know blueberry are a low-calorie, nutrient-rich fruit? They are full of antioxidants and essential nutrients like vitamin C, vitamin K and manganese. Not so blue after hearing this right =)",
        6:  "Did you know bottle gourd improves digestive health and aids in weight loss? They contain 0 fat and many nutrients such as potassium, zinc, magnesium and vitamin C. ",
        7:  "Did you know brinjal is a low-calorie, naturally fat-free source of complex carbohydrates with plenty of fiber? Definitely a heart-healthy addition to a balanced diet. ",
        8:  "Did you know broccoli is considered to be one of the most nutritious vegetables? It is low in calories and sodium, fat-free and cholesterol-free. If Bruce Lee had a brother, it would be Broco-Lee =)",
        9:  "Did you know cabbage is a low-calorie, nearly fat-free food that is a good source of potassium, folate, and vitamin K? Half of cabbage's carbs comes from fiber. Time to chomp down on more kimchi!!!",
        10: "Did you know capsicum are a low-calorie, low-fat source of carbohydrates, including fiber, as well as many nutrients such as vitamin C, vitamin A, potassium, magnesium, zinc, and vitamin E? ",
        11: "Did you know carrot are a healthy source of carbohydrates and fiber while being low in fat, protein, and sodium? Carrots are also high in vitamin A and nutrients. No wonder rabbits love them!",
        12: "Did you know cauliflower is a fiber-rich vegetable that is low in fat and calories? It is a great source of vitamin C , vitamin B6 and magnesium. Time to replace your white rice with cauliflower rice!",
        13: "Did you know coconut is rich in plant-based saturated fats that may offer health benefits? It is an excellent source of manganese and other minerals. Time to order more Mr Cococnut milkshakes haha",
        14: "Did you know cucumber is a low-calorie food that is primarily water? It also provides nutrients (potassium, vitamins K and C). Remember to add some cucumber in your Subway sandwich when ordering!",
        15: "Did you know dragon Fruit contains vitamin C, magnesium and is packed with healthy fiber? Eating it would not turn you into a dragon but do include it in your daily salad bowl for more benefits.",
        16: "Did you know durian is called the 'King of Fruits' because of its intensely pungent smell? Durian contains a fair amount of fiber and protein. Try not to get injured from handling the thorny fruit.",
        17: "Did you know grapes are a vitamin-rich and hydrating fruit that provides plenty of vitamin C, K, and A? Another reason for you to drink more Ribena. Sounds Grape(great) right? =)",
        18: "Did you know guava is a tropical fruit that contains an excellent source of vitamin C, vitamin A and folate? Guava helps to promote skin health and aids cell protection and repair in your body.",
        19: "Did you know kiwis are an excellent source of complex carbohydrates, offering fiber and antioxidants? Kiwis also provide more than your daily requirements of vitamin C and plenty of vitamin K. ",
        20: "Did you know lemons are an excellent source of vitamin C, low in calorie and relatively high in fiber? They provide other vitamins and minerals too. When life gives you lemons, eat it!!!",
        21: "Did you know limes are a great source of vitamin C, phytonutrients and antioxidant properties? Don't forget to order some lime juice when you're at a mamak stall!",
        22: "Did you know longan is also called the Dragon Eye Fruit due to its appearance (Peeled, the fruit looks like an eyeball)? It is a great source of vitamin C, potassium and provides a boost of fiber. ",
        23: "Did you know mangoes are a nutrient rich source of carbohydrates, packed with vitamin C? They are low in fat, sodium, cholesterol, and contain lots of vitamins and minerals. This is mango-nificent!",
        24: "Did you know muskmelon is a nutrient-dense source of carbohydrates? The melon is a rich source of vitamin C, A, and potassium. It contains magnesium, vitamin K, zinc, and folate too.",
        25: "Did you know oranges are fiber-rich fruit that provides tons of vitamin C and potassium? A serving contains more than a day's worth of vitamin C. Do not reject your Mum if she gives you Oh-Leng to eat.",
        26: "Did you know papaya is a fat-free, nutrient-rich source of healthy carbohydrates, including fiber? It provides plenty of vitamin C, with 98% of your daily recommended intake. Papayeah!!!",
        27: "Did you know passion fruit is a good source of fiber and protein? You also get a healthy dose of vitamin and nutrients when you consume this fruit. Time to cultivate some passion for this fruit.",
        28: "Did you know pears are a high-fiber source of carbohydrates that provide a low-calorie burst of vitamin C as well as minerals? They are pear-fect for snacking when you're hungry.",
        29: "Did you know pineapple is a good source of vitamin C while being low in fat and sodium with an abundance of minerals? Be like a Pineapple, stand tall, Wear a crown and be sweet on the inside!",
        30: "Did you know plums are a low-fat, low-calorie, high-fiber source of carbohydrates? Plums contain lots of antioxidants and some vitamin C, vitamin A, vitamin K, copper, and manganese. ",
        31: "Did you know pomegranate is a lower-calorie, very low-fat, nutrient-dense food providing a large amount of fiber? The fruit is also an excellent source of potassium, magnesium, vitamin C, and zinc.",
        32: "Did you know pomelos are a fiber-rich fruit that is packed with vitamin C and potassium? They also contain some other vitamins and minerals and are low in fat, cholesterol, and sodium.",
        33: "Did you know potatoes are a good source of potassium, vitamin C and vitamin B6? Potassium, which works in opposition to sodium, helps regulate blood pressure. Sounds spud-tacular right!!",
        34: "Did you know pumpkin is fairly low in calories and fat? The carbs it contains are a mixture of fiber, naturally occurring sugars, and starch. Pumpkin is an excellent source of vitamins and minerals.",
        35: "Did you know radish is low-calorie, low-sodium, fat-free and cholesterol-free? It contains 19% of the daily recommended vitamin C value. Time to order some popiah at your hawker centre.",
        36: "Did you know strawberries are a fiber-rich source of complex carbs that is naturally low in calories and fat? It is a good source of vitamins and minerals. Strawberry ice cream does not count though =(",
        37: "Did you know tomatoes are a low-calorie, low-fat hydrating fruit with a low glycemic index? Tomatoes are high in vitamin C, vitamin K, and potassium. Maybe it's time to ketchup with Tomato =)",
        38: "Did you know watermelon is low in calories and contains almost no fat? It provides many valuable nutrients (vitamins A and C) and is high in antioxidants. Be a Melon-aire by eating Watermelon!",
    }
    
    # img_ = image.load_img(userimage, target_size=(150, 150))
    img_array = image.img_to_array(userimage)
    img_processed = np.expand_dims(img_array, axis=0) 
    img_processed /= 255.   
    
    prediction = model.predict(img_processed)
    index = np.argmax(prediction)
    value = prediction[0][index]

    return category[index], value, writeup[index], thumbnail[index]











































