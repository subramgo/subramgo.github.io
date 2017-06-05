---
layout: post
title: Python List Insertion - Preserve the sort order
comments: true
---

Let us say we have the following python snippet, where we create a list and sort the same.

```
>>> a_list = [10,1,34,56,2]
>>> a_list = sorted(a_list)
>>> a_list
[1, 2, 10, 34, 56]

```
<!--break-->

Let us now add another element 45 to list, to the death of our sorting.

```
>>> a_list.append(45)
>>> a_list
[1, 2, 10, 34, 56, 45]

```

How do we maintain the sort order in our list as we add new elements to it ? Module *bisect* comes to our rescue.

```
>>> a_list = [10,1,34,56,2]
>>> a_list = sorted(a_list)
>>> a_list
[1, 2, 10, 34, 56]

>>> import bisect
>>> idx = bisect.bisect_left(a_list, 45)
>>> idx
4
>>> a_list.insert(idx,45)
>>> a_list
[1, 2, 10, 34, 45, 56]


```
As you can see the function **bisect_left** returned the position where we need to insert our new element. Our new elements goes to position 4 in the list and we finally use the list insert function to insert 45 into 4th position.

What is the significance of **_left** ? If 45 is already present in the list, it will be inserted to the left of existing 45 entry. We also have another function called **bisect_right**. This will insert the new element to the right of the existing element.


We also have a convenient function which combines both **bisect_left** and list.insert, **insort_left**.

```
>>> bisect.insort_left(a_list,12)
>>> a_list
[1, 2, 10, 12, 34, 45, 56]
>>> 

```


[module bisect](https://docs.python.org/2.7/library/bisect.html#module-bisect)

{% include comments.html %}

