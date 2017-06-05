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