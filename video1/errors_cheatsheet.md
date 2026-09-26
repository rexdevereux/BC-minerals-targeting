# Common Python errors and what they mean

Errors look scary. They're not. They tell you exactly what went wrong, where, and usually what to do about it. Here are the ones you'll hit in your first hour with Python.

## How to read any error

Python errors have three parts:

1. **The traceback** — a list of where the error happened (read it bottom-up)
2. **The error type** — one or two words, like `SyntaxError` or `NameError`
3. **The message** — a short explanation of what went wrong

The error type tells you the *category*. The message tells you the *specific*. Always read both.

## SyntaxError

You typed something Python can't parse. Almost always a typo — missing quote, missing parenthesis, missing colon.

```python
print("hello)
# SyntaxError: unterminated string literal
```

Fix: count your quotes and brackets.

## NameError

You used a name Python doesn't recognize. Usually a typo, or you forgot to define the variable, or you forgot to import something.

```python
print(grade_gpt)
# NameError: name 'grade_gpt' is not defined
```

Fix: check spelling, check that you ran the cell that defined the variable.

## TypeError

You tried to do something to a value that doesn't make sense for its type. Classic example: adding a number to a string.

```python
"DDH-" + 1
# TypeError: can only concatenate str (not "int") to str
```

Fix: convert the type, e.g. `"DDH-" + str(1)`.

## IndentationError

Python uses indentation (spaces at the start of a line) to group code. Mix tabs and spaces, or get the indentation wrong, and you get this.

```python
if grade > 1.0:
print("ore grade")
# IndentationError: expected an indented block
```

Fix: indent the line with 4 spaces.

## FileNotFoundError

You asked Python to open a file that isn't where you said it is.

```python
df = pd.read_csv("assays.csv")
# FileNotFoundError: [Errno 2] No such file or directory: 'assays.csv'
```

Fix in Colab: make sure you uploaded the file using the folder icon on the left sidebar, and that the name matches exactly (case-sensitive).

## KeyError

You asked for a column or dictionary key that doesn't exist.

```python
df["Gold"]
# KeyError: 'Gold'
```

Fix: check the actual column names with `df.columns`. Probably it's `Au_gpt`, not `Gold`.

## When in doubt

Copy the last line of the error and paste it into Google or ChatGPT. Someone has hit this before. The Python community is enormous and welcoming — error messages are a shared language.
