### 🐍 **Beginner Level (1-15)**

1. **What is Python? What are its key features?**  
   Python is a high-level, interpreted, general-purpose programming language known for its readability.  
   **Key features**: Interpreted, dynamically typed, object-oriented, large standard library, open-source.

2. **What are Python’s data types?**  
   - Numeric: `int`, `float`, `complex`  
   - Sequence: `list`, `tuple`, `range`  
   - Text: `str`  
   - Set: `set`, `frozenset`  
   - Mapping: `dict`  
   - Boolean: `bool`  
   - Binary: `bytes`, `bytearray`, `memoryview`

3. **What is the difference between a list and a tuple?**  
   - `list` is mutable (can be changed).  
   - `tuple` is immutable (cannot be changed).  
   - Lists consume more memory than tuples.

4. **How is memory managed in Python?**  
   Python uses **automatic memory management**:  
   - Reference counting  
   - Garbage collection  
   - Object pooling (for small integers and strings)

5. **What are Python's conditional statements?**  
   `if`, `elif`, and `else` are used for decision-making.

6. **How does a `for` loop differ from a `while` loop?**  
   - `for`: loops over a sequence or iterator  
   - `while`: loops while a condition is true

7. **What is indentation in Python? Why is it important?**  
   Python uses **indentation to define code blocks** instead of curly braces. It's mandatory.

8. **What are functions in Python?**  
   A function is a reusable block of code. Declared with `def`.

9. **What is the difference between `is` and `==`?**  
   - `==`: compares **values**  
   - `is`: compares **identities (memory address)**

10. **What are *args and **kwargs?**  
   - `*args`: non-keyword variable arguments  
   - `**kwargs`: keyword arguments (dictionary)

11. **How do you handle exceptions in Python?**  
   Use `try`, `except`, `finally`, `else`.  
   Example:
   ```python
   try:
       1 / 0
   except ZeroDivisionError:
       print("Cannot divide by zero")
   ```

12. **What is a dictionary in Python?**  
   A key-value mapping. Example:
   ```python
   {"name": "Alice", "age": 30}
   ```

13. **What are Python’s boolean values?**  
   `True` and `False` (with capital T and F)

14. **What is list comprehension?**  
   A concise way to create lists:
   ```python
   squares = [x**2 for x in range(5)]
   ```

15. **What is the difference between `append()` and `extend()` in lists?**  
   - `append()`: adds a single element  
   - `extend()`: adds elements from another iterable

---

### 🐍 **Intermediate Level (16–35)**

16. **What is a lambda function?**  
   Anonymous inline function.  
   Example:
   ```python
   f = lambda x: x + 1
   ```

17. **What are Python modules and packages?**  
   - Module: A Python file  
   - Package: A directory with `__init__.py` that contains modules

18. **What is a decorator?**  
   A function that modifies another function.  
   Example:
   ```python
   def decorator(fn):
       def wrapper():
           print("Before")
           fn()
           print("After")
       return wrapper
   ```

19. **Explain Python’s object-oriented features.**  
   Python supports classes, objects, inheritance, polymorphism, encapsulation.

20. **What is `self` in Python classes?**  
   Refers to the **instance** of the class. It must be the first parameter of instance methods.

21. **What is the difference between a class and an object?**  
   - Class: blueprint  
   - Object: instance of a class

22. **What is `__init__()`?**  
   Constructor method. Called when an object is created.

23. **What are instance, class, and static methods?**  
   - `@staticmethod`: No self/cls, like a normal function  
   - `@classmethod`: First argument is `cls`  
   - Instance methods: First argument is `self`

24. **What is the difference between shallow and deep copy?**  
   - Shallow: references nested objects  
   - Deep: copies nested objects too  
   Use `copy` and `deepcopy` from `copy` module.

25. **What is a generator?**  
   A function that returns an iterator using `yield`.  
   Example:
   ```python
   def gen():
       yield 1
       yield 2
   ```

26. **What is the use of `with` statement?**  
   Context manager, auto-handles cleanup.  
   Example:
   ```python
   with open('file.txt') as f:
       data = f.read()
   ```

27. **How are exceptions raised manually?**  
   Use `raise`:
   ```python
   raise ValueError("Invalid input")
   ```

28. **What are the different ways to import a module?**  
   ```python
   import math  
   from math import pi  
   import math as m
   ```

29. **What are Python’s scopes?**  
   LEGB Rule:  
   - Local  
   - Enclosing  
   - Global  
   - Built-in

30. **What is recursion?**  
   A function calling itself. Must have a base case to prevent infinite calls.

31. **What are Python’s built-in data structures?**  
   - `list`, `tuple`, `set`, `dict`

32. **What is duck typing in Python?**  
   “If it walks like a duck and quacks like a duck...”  
   Behavior matters more than type.

33. **What is a Python virtual environment?**  
   An isolated environment to manage dependencies per project. Created with `venv`.

34. **What is PEP 8?**  
   Python Enhancement Proposal – style guide for Python code.

35. **What is the difference between `@staticmethod` and `@classmethod`?**  
   - `staticmethod`: no access to class or instance  
   - `classmethod`: accesses class (`cls`)

---

### 🐍 **Advanced Level (36–50)**

36. **Explain Python's GIL (Global Interpreter Lock).**  
   A mutex that allows only one thread to execute at a time in CPython, limiting true parallelism in multithreading.

37. **How does multithreading differ from multiprocessing?**  
   - `threading`: shared memory, lightweight  
   - `multiprocessing`: separate memory, better for CPU-bound tasks

38. **What are Python’s built-in functions for functional programming?**  
   `map()`, `filter()`, `reduce()`, `zip()`

39. **What is the difference between `isinstance()` and `type()`?**  
   - `type()` returns the type  
   - `isinstance()` checks if an object is an instance of a class or subclass

40. **What is the difference between `==` and `__eq__()`?**  
   - `==` uses `__eq__()` under the hood.  
   You can override `__eq__()` in custom classes.

41. **What are metaclasses in Python?**  
   Classes of classes. They define the behavior of class creation using `type`.

42. **What is monkey patching?**  
   Modifying or extending code at runtime without altering the original source.

43. **How do you perform memory profiling in Python?**  
   Use modules like `memory_profiler`, `tracemalloc`.

44. **How to optimize Python code for performance?**  
   - Use generators  
   - Use built-in functions  
   - Avoid unnecessary loops  
   - Use Cython or NumPy for heavy computations

45. **What are coroutines and `async/await`?**  
   Used for asynchronous programming.  
   ```python
   async def foo():
       await bar()
   ```

46. **What is `__slots__` in Python?**  
   Optimizes memory usage by preventing dynamic creation of attributes.

47. **What are dunder (magic) methods?**  
   Special methods with double underscores.  
   Example: `__init__`, `__str__`, `__len__`

48. **What is the use of the `inspect` module?**  
   Helps retrieve info about live objects: classes, functions, source code, etc.

49. **What is a memory leak in Python?**  
   When objects are not released properly, often due to circular references or global references.

50. **How is exception chaining handled in Python?**  
   Using `raise ... from ...` to keep track of the original exception.