#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 08:15:38 2020

https://www.youtube.com/watch?v=HGOBQPFzWKo&list=WL&index=37&t=61s

@author: sean
"""

my_list = ['string', 'another string', 'a third one']
print(my_list)

example_list = [5,True, "apple","apple"]
item = my_list[0]
print(item)
print(my_list[-1])
print(my_list[-2])

print(len(my_list))

my_list.insert(1,"blueberry")
print(my_list)

slicing = [1,2,3,4,5,6,7,8,9]
sliced_list = slicing[1:5]
print(sliced_list)

b = [i*i for i in slicing]
print(b)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

my_tuple = ("max",28, "boston")
print(type(my_tuple))

print(my_tuple.index("boston"))

name, age, city = my_tuple
print(name,age,city)

new_tuple = (0,1,2,3,4)
i1, *i2, i3  = new_tuple
print(i1,i3,i2)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

my_dict = {"name": "Max", "age": 28, "city": "New York"}
print(my_dict)
my_dict2= dict(name="mary", age = 27, city = "Boston")
print(my_dict2)

my_dict["email"] = "string@string.com"
print(my_dict)

del my_dict["name"]
print(my_dict)

my_dict = {"name": "Max", "age": 28, "city": "New York"}

if "name" in my_dict:
    print(my_dict["name"])
    

my_dict = {"name": "Max", "age": 28, "email": "New York"}
my_dict.update(my_dict2)

print(my_dict)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
my_set = set("hello")
print(my_set)

my_set.discard("e") # removes element if it exists, otherwise does nothing

odds = {1,3,5,7,9}
evens = {2,4,6,8,10}
primes = {2,3,5,7}

union = odds.union(evens)
print(union)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

my_string = '      hello       '
print(my_string)
my_string = my_string.strip()
print(my_string)

print(my_string.upper())
print(my_string.lower())

my_string = "hello world"
print(my_string.replace("world", "universe"))
