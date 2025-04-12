### 🔥 **Advanced Python Interview Questions (Bonus Set)**

---

**1. What are closures in Python? How are they used?**  
**Answer:**  
A **closure** is a function object that has access to variables in its **lexical scope**, even when the function is called outside that scope.  
Closures are created when:
- A nested function references a value from its enclosing scope.
- The enclosing function returns the nested function.

✅ **Example:**
```python
def outer(msg):
    def inner():
        print(msg)  # `msg` is captured in closure
    return inner

fn = outer("Hello")
fn()  # prints "Hello"
```

Use cases: decorators, data hiding, callbacks.

---

**2. What is the difference between deep and shallow copies in custom classes?**  
**Answer:**  
- `shallow copy` (`copy.copy()`): copies outer object, not nested ones (they’re shared).
- `deep copy` (`copy.deepcopy()`): recursively copies all objects.

✅ **Custom class example:**
```python
import copy

class Person:
    def __init__(self, name, hobbies):
        self.name = name
        self.hobbies = hobbies

p1 = Person("Alice", ["reading", "chess"])
p2 = copy.deepcopy(p1)
p2.hobbies.append("swimming")
```

Without deep copy, `p1` and `p2` would share the same `hobbies` list.

---

**3. How is memory optimization achieved with `__slots__`?**  
**Answer:**  
Using `__slots__` tells Python not to use a dynamic `__dict__` for attribute storage, reducing memory usage.

✅ **Example:**
```python
class Point:
    __slots__ = ('x', 'y')

    def __init__(self, x, y):
        self.x = x
        self.y = y
```

⚠️ Limitation: You cannot add attributes beyond those defined in `__slots__`.

---

**4. What is the difference between `@property` and regular methods?**  
**Answer:**  
`@property` allows method access like an attribute and is used for encapsulation.

✅ **Example:**
```python
class Temperature:
    def __init__(self, celsius):
        self._celsius = celsius

    @property
    def fahrenheit(self):
        return self._celsius * 9/5 + 32
```

You can now call `temp.fahrenheit` without parentheses.

---

**5. How does Python handle memory management and garbage collection?**  
**Answer:**  
Python uses:
- **Reference counting**
- **Garbage collector (GC)** for cyclic references  
   `gc` module provides manual control.

✅ **Check reference count:**
```python
import sys
a = []
print(sys.getrefcount(a))  # includes the temporary reference
```

---

**6. What is the purpose of `__enter__` and `__exit__` methods?**  
**Answer:**  
They are used in context managers (`with` statements).  
`__enter__()` runs on entering the context, `__exit__()` on exit.

✅ **Example:**
```python
class FileOpener:
    def __init__(self, filename):
        self.filename = filename

    def __enter__(self):
        self.file = open(self.filename)
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()
```

---

**7. How do you create a custom iterable object?**  
**Answer:**  
Implement `__iter__()` and `__next__()` in a class.

✅ **Example:**
```python
class Counter:
    def __init__(self, limit):
        self.limit = limit
        self.count = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.count >= self.limit:
            raise StopIteration
        self.count += 1
        return self.count
```

---

**8. What are Python descriptors and how do they work?**  
**Answer:**  
Descriptors define behavior for attribute access using methods like:
- `__get__`, `__set__`, `__delete__`

Used internally by `property`, `staticmethod`, etc.

✅ **Example:**
```python
class MyDescriptor:
    def __get__(self, obj, objtype=None):
        return "Got value"

class MyClass:
    attr = MyDescriptor()

obj = MyClass()
print(obj.attr)  # "Got value"
```

---

**9. What are the differences between coroutines, generators, and async generators?**  
**Answer:**
- **Generator:** yields values using `yield`
- **Coroutine:** waits using `await`, defined with `async def`
- **Async generator:** combination of both: `async def` + `yield`

✅ **Async generator:**
```python
async def my_gen():
    for i in range(3):
        yield i
```

---

**10. How does Python's `dataclass` differ from regular classes?**  
**Answer:**  
`dataclasses` auto-generate methods like `__init__`, `__repr__`, `__eq__`.

✅ **Example:**
```python
from dataclasses import dataclass

@dataclass
class Point:
    x: int
    y: int
```

Benefits:
- Cleaner syntax
- Immutability via `frozen=True`
- Field customization with `field()`