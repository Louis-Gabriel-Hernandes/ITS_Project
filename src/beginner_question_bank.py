from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .utils import ensure_dir, write_json

BEGINNER_TOPICS: list[tuple[str, str, str]] = [
    ("variables", "variable_assignment", "storing and updating values"),
    ("primitive_types", "primitive_type_selection", "choosing int, double, boolean, and char"),
    ("strings", "string_basics", "using text values and simple String methods"),
    ("operators", "operator_basics", "using arithmetic, comparison, and boolean operators"),
    ("conditionals", "conditional_logic", "choosing the branch that runs"),
    ("loops", "loop_boundaries", "counting loop iterations safely"),
    ("arrays", "array_indexing", "accessing and traversing arrays"),
    ("methods", "method_parameters", "calling methods, parameters, and return values"),
    ("classes_objects", "object_state", "creating simple objects and reading fields"),
    ("debugging", "syntax_vs_logic_error", "finding small beginner Java mistakes"),
    ("code_tracing", "trace_execution", "predicting output one line at a time"),
]

QUESTION_TYPES = ["multiple_choice", "fill_blank", "code_tracing", "debugging", "short_code"]
NAMES = ["score", "total", "age", "count", "level", "points", "coins", "minutes", "tries", "speed"]
STRINGS = ["Java", "Code", "Blue", "Cat", "Book", "Robot", "Loop", "Array", "Class", "Input"]


def _hints(topic: str, skill: str, qtype: str) -> list[str]:
    topic_name = topic.replace("_", " ")
    return [
        f"Focus on the {topic_name} idea before calculating or choosing.",
        "Write down the value of each variable after the line runs.",
        f"Check the Java syntax for {skill.replace('_', ' ')} and try the smallest example.",
    ]


def _q(
    qid: str,
    topic: str,
    skill: str,
    difficulty: int,
    qtype: str,
    prompt: str,
    correct: str,
    explanation: str,
    *,
    choices: list[str] | None = None,
    accepted: list[str] | None = None,
    patterns: list[str] | None = None,
    tags: list[str] | None = None,
) -> Dict[str, Any]:
    return {
        "question_id": qid,
        "topic": topic,
        "skill": skill,
        "difficulty": difficulty,
        "question_type": qtype,
        "prompt": prompt,
        "choices": choices or [],
        "correct_answer": correct,
        "accepted_answers": accepted or [],
        "answer_patterns": patterns or [],
        "explanation": explanation,
        "misconception_tags": tags or [skill],
        "hint_ids": tags or [skill],
        "hints": _hints(topic, skill, qtype),
        "estimated_time_seconds": 45 + difficulty * 25,
    }


def _shuffle_choices(correct: str, wrong: list[str], seed: int) -> list[str]:
    # Deterministic small rotation without importing random; enough to avoid same answer position.
    choices = [correct, *wrong]
    shift = seed % len(choices)
    return choices[shift:] + choices[:shift]


def _variables(qid: str, difficulty: int, variant: int, qtype: str) -> Dict[str, Any]:
    name = NAMES[variant % len(NAMES)]
    a = difficulty + variant + 2
    b = a + difficulty
    skill = "variable_assignment"
    if qtype == "multiple_choice":
        correct = f"int {name} = {a};"
        return _q(qid, "variables", skill, difficulty, qtype,
                  f"Which line correctly creates an integer variable named `{name}` with the value {a}?",
                  correct,
                  "A Java integer variable can be declared with `int`, followed by the name, equals sign, value, and semicolon.",
                  choices=_shuffle_choices(correct, [f"integer {name} = {a};", f"int = {name} {a};", f"{name} int = {a}"], variant),
                  tags=["variable_assignment", "primitive_type_selection"])
    if qtype == "fill_blank":
        return _q(qid, "variables", skill, difficulty, qtype,
                  f"Fill in the blank to update `{name}` to {b}: `{name} _____ {b};`",
                  "=",
                  "Assignment uses a single equals sign to store a new value in an existing variable.",
                  accepted=["equals"], tags=["variable_assignment"])
    if qtype == "code_tracing":
        return _q(qid, "variables", skill, difficulty, qtype,
                  f"What is printed?\n```java\nint {name} = {a};\n{name} = {name} + {difficulty};\nSystem.out.println({name});\n```",
                  str(a + difficulty),
                  f"The variable starts at {a}, then increases by {difficulty}, so it prints {a + difficulty}.",
                  tags=["variable_assignment", "trace_execution"])
    if qtype == "debugging":
        return _q(qid, "variables", skill, difficulty, qtype,
                  f"Find the bug in this line: `int {name} {a};`",
                  "missing equals sign",
                  "A declaration with an initial value needs `=` between the variable name and the value.",
                  accepted=["add =", "missing =", "equals sign"], patterns=["="], tags=["variable_assignment", "syntax_vs_logic_error"])
    return _q(qid, "variables", skill, difficulty, qtype,
              f"Write one Java statement that declares an int named `{name}` and stores {a} in it.",
              f"int {name} = {a};",
              "The statement needs a type, variable name, assignment, value, and semicolon.",
              accepted=[f"int {name}={a};"], patterns=["int", name, str(a)], tags=["variable_assignment"])


def _primitive_types(qid: str, difficulty: int, variant: int, qtype: str) -> Dict[str, Any]:
    skill = "primitive_type_selection"
    examples = [
        ("whole number of students", "int", "27"),
        ("price with cents", "double", "4.99"),
        ("true or false value", "boolean", "true"),
        ("one letter grade", "char", "'A'"),
    ]
    label, typ, value = examples[variant % len(examples)]
    var = NAMES[(variant + 2) % len(NAMES)]
    if qtype == "multiple_choice":
        return _q(qid, "primitive_types", skill, difficulty, qtype,
                  f"Which Java type is best for storing a {label}?",
                  typ,
                  f"`{typ}` is the best match for this kind of value.",
                  choices=_shuffle_choices(typ, [t for _, t, _ in examples if t != typ][:3], variant), tags=[skill])
    if qtype == "fill_blank":
        return _q(qid, "primitive_types", skill, difficulty, qtype,
                  f"Fill in the blank: `_____ {var} = {value};`",
                  typ,
                  f"The value `{value}` matches the Java type `{typ}`.",
                  tags=[skill])
    if qtype == "code_tracing":
        n = 10 + variant + difficulty
        return _q(qid, "primitive_types", skill, difficulty, qtype,
                  f"What is printed?\n```java\nint whole = {n};\ndouble decimal = whole / 2.0;\nSystem.out.println(decimal);\n```",
                  str(n / 2.0),
                  "Dividing by `2.0` uses decimal arithmetic and stores the result in a double.",
                  accepted=[str(n / 2)], tags=[skill, "trace_execution"])
    if qtype == "debugging":
        return _q(qid, "primitive_types", skill, difficulty, qtype,
                  f"Find the type bug: `int {var} = {4.5 + variant};`",
                  "use double",
                  "A decimal value should be stored in a `double`, not an `int`.",
                  accepted=["double", "change int to double"], patterns=["double"], tags=[skill, "syntax_vs_logic_error"])
    return _q(qid, "primitive_types", skill, difficulty, qtype,
              f"Write a Java statement that stores `{value}` in a variable named `{var}` using the best primitive type.",
              f"{typ} {var} = {value};",
              f"The value `{value}` matches `{typ}`.",
              patterns=[typ, var], tags=[skill])


def _strings(qid: str, difficulty: int, variant: int, qtype: str) -> Dict[str, Any]:
    skill = "string_basics"
    word = STRINGS[variant % len(STRINGS)]
    other = STRINGS[(variant + 3) % len(STRINGS)]
    if qtype == "multiple_choice":
        correct = "name.equals(\"Java\")"
        return _q(qid, "strings", skill, difficulty, qtype,
                  "Which expression checks whether the String variable `name` contains the text `Java`?",
                  correct,
                  "Use `.equals(...)` to compare String contents in Java.",
                  choices=_shuffle_choices(correct, ["name == \"Java\"", "name = \"Java\"", "name.equals = \"Java\""], variant),
                  tags=["string_comparison"])
    if qtype == "fill_blank":
        return _q(qid, "strings", skill, difficulty, qtype,
                  "Fill in the blank to get the number of characters in `word`: `word._____();`",
                  "length",
                  "A String uses the `length()` method to count its characters.",
                  accepted=["length()"], tags=[skill])
    if qtype == "code_tracing":
        return _q(qid, "strings", skill, difficulty, qtype,
                  f"What is printed?\n```java\nString first = \"{word}\";\nString second = \"{other}\";\nSystem.out.println(first + second);\n```",
                  word + other,
                  "The `+` operator joins two strings together in order.",
                  tags=[skill, "trace_execution"])
    if qtype == "debugging":
        return _q(qid, "strings", skill, difficulty, qtype,
                  f"Find the bug: `if (word == \"{word}\") {{ System.out.println(word); }}`",
                  "use equals",
                  "For beginner Java String comparisons, use `.equals(...)` instead of `==`.",
                  accepted=[".equals", f"word.equals(\"{word}\")", "use .equals"], patterns=["equals"], tags=["string_comparison"])
    return _q(qid, "strings", skill, difficulty, qtype,
              f"Write one Java statement that creates a String variable named `word` with the value `{word}`.",
              f"String word = \"{word}\";",
              "String literals use double quotes.",
              patterns=["String", "word", word], tags=[skill])


def _operators(qid: str, difficulty: int, variant: int, qtype: str) -> Dict[str, Any]:
    skill = "operator_basics"
    a = 3 + variant + difficulty
    b = 2 + (variant % 4)
    if qtype == "multiple_choice":
        correct = "%"
        return _q(qid, "operators", skill, difficulty, qtype,
                  "Which operator gives the remainder after division in Java?",
                  correct,
                  "The `%` operator gives the remainder, such as `7 % 3` giving 1.",
                  choices=_shuffle_choices(correct, ["/", "*", "&&"], variant), tags=["operator_precedence"])
    if qtype == "fill_blank":
        return _q(qid, "operators", skill, difficulty, qtype,
                  f"Fill in the blank so the condition checks equality: `if ({a} _____ {a})`",
                  "==",
                  "Java uses `==` for primitive equality checks.",
                  accepted=["equals equals"], tags=["operator_precedence"])
    if qtype == "code_tracing":
        result = a + b * difficulty
        return _q(qid, "operators", skill, difficulty, qtype,
                  f"What is printed?\n```java\nSystem.out.println({a} + {b} * {difficulty});\n```",
                  str(result),
                  "Multiplication happens before addition unless parentheses change the order.",
                  tags=["operator_precedence", "trace_execution"])
    if qtype == "debugging":
        return _q(qid, "operators", skill, difficulty, qtype,
                  f"Find the bug: `if (count = {a}) {{ System.out.println(count); }}`",
                  "use ==",
                  "A condition should compare with `==`; a single `=` is assignment.",
                  accepted=["==", "count ==", "use equals equals"], patterns=["=="], tags=["operator_precedence", "syntax_vs_logic_error"])
    return _q(qid, "operators", skill, difficulty, qtype,
              f"Write a Java expression that adds `{a}` and `{b}`, then multiplies the result by {difficulty}.",
              f"({a} + {b}) * {difficulty}",
              "Parentheses make the addition happen before multiplication.",
              patterns=["(", "+", ")", "*"], tags=["operator_precedence"])


def _conditionals(qid: str, difficulty: int, variant: int, qtype: str) -> Dict[str, Any]:
    skill = "conditional_logic"
    x = 4 + variant + difficulty
    threshold = x - (variant % 3 + 1)
    if qtype == "multiple_choice":
        correct = "The if block runs"
        return _q(qid, "conditionals", skill, difficulty, qtype,
                  f"If `x` is {x}, what happens in `if (x > {threshold})`?",
                  correct,
                  f"{x} is greater than {threshold}, so the `if` condition is true.",
                  choices=_shuffle_choices(correct, ["The else block runs", "Both blocks run", "The code cannot compile"], variant), tags=[skill])
    if qtype == "fill_blank":
        return _q(qid, "conditionals", skill, difficulty, qtype,
                  f"Fill in the blank to check if `age` is at least {x}: `if (age _____ {x})`",
                  ">=",
                  "`>=` means greater than or equal to.",
                  accepted=["greater than or equal", "greater than or equal to"], tags=[skill])
    if qtype == "code_tracing":
        answer = "big" if x > 10 else "small"
        return _q(qid, "conditionals", skill, difficulty, qtype,
                  f"What is printed?\n```java\nint x = {x};\nif (x > 10) {{\n    System.out.println(\"big\");\n}} else {{\n    System.out.println(\"small\");\n}}\n```",
                  answer,
                  "Check the condition first, then follow only the branch that matches.",
                  tags=[skill, "trace_execution"])
    if qtype == "debugging":
        return _q(qid, "conditionals", skill, difficulty, qtype,
                  "Find the bug: `if (isReady = true) { System.out.println(\"Go\"); }`",
                  "use == or just isReady",
                  "A condition should not assign a value by mistake. Use `isReady == true` or simply `isReady`.",
                  accepted=["==", "isReady", "use equals equals"], patterns=["="], tags=[skill, "syntax_vs_logic_error"])
    return _q(qid, "conditionals", skill, difficulty, qtype,
              f"Write an if statement header that checks whether `score` is greater than or equal to {x}.",
              f"if (score >= {x})",
              "An if header starts with `if`, uses parentheses, and contains the boolean condition.",
              patterns=["if", "score", ">=", str(x)], tags=[skill])


def _loops(qid: str, difficulty: int, variant: int, qtype: str) -> Dict[str, Any]:
    skill = "loop_boundaries"
    n = 3 + difficulty + (variant % 5)
    if qtype == "multiple_choice":
        correct = str(n)
        return _q(qid, "loops", skill, difficulty, qtype,
                  f"How many times does this loop run? `for (int i = 0; i < {n}; i++)`",
                  correct,
                  f"The loop runs for i = 0 through {n - 1}, which is {n} times.",
                  choices=_shuffle_choices(correct, [str(n - 1), str(n + 1), "0"], variant), tags=[skill, "off_by_one"])
    if qtype == "fill_blank":
        return _q(qid, "loops", skill, difficulty, qtype,
                  f"Fill in the blank to loop while `i` is less than {n}: `for (int i = 0; i _____ {n}; i++)`",
                  "<",
                  "Using `<` stops before the limit, which is common for counting from zero.",
                  accepted=["less than"], tags=[skill])
    if qtype == "code_tracing":
        total = sum(range(n))
        return _q(qid, "loops", skill, difficulty, qtype,
                  f"What is printed?\n```java\nint total = 0;\nfor (int i = 0; i < {n}; i++) {{\n    total += i;\n}}\nSystem.out.println(total);\n```",
                  str(total),
                  f"The loop adds 0 through {n - 1}; the total is {total}.",
                  tags=[skill, "trace_execution"])
    if qtype == "debugging":
        return _q(qid, "loops", skill, difficulty, qtype,
                  f"Find the boundary bug: `for (int i = 0; i <= {n}; i++) System.out.println(i);` when you want exactly {n} values starting at 0.",
                  "use i < n",
                  "Use `<` when counting exactly n values from 0 to n - 1.",
                  accepted=["i <", "change <= to <", "off by one"], patterns=["<"], tags=[skill, "off_by_one"])
    return _q(qid, "loops", skill, difficulty, qtype,
              f"Write a for-loop header that counts `i` from 0 up to but not including {n}.",
              f"for (int i = 0; i < {n}; i++)",
              "The standard beginner loop starts at 0, checks `< limit`, and increments with `i++`.",
              patterns=["for", "int i", "0", "<", str(n), "i++"], tags=[skill])


def _arrays(qid: str, difficulty: int, variant: int, qtype: str) -> Dict[str, Any]:
    skill = "array_indexing"
    n = 4 + difficulty + (variant % 4)
    idx = variant % n
    values = [idx + i + 1 for i in range(4)]
    if qtype == "multiple_choice":
        correct = str(n - 1)
        return _q(qid, "arrays", skill, difficulty, qtype,
                  f"An array has length {n}. What is its largest valid index?",
                  correct,
                  "Java array indexes start at 0, so the final index is length - 1.",
                  choices=_shuffle_choices(correct, [str(n), "1", str(n + 1)], variant), tags=[skill, "off_by_one"])
    if qtype == "fill_blank":
        return _q(qid, "arrays", skill, difficulty, qtype,
                  "Fill in the blank to get the length of array `nums`: `nums._____`",
                  "length",
                  "Arrays use the `.length` field, not a `length()` method.",
                  tags=[skill])
    if qtype == "code_tracing":
        return _q(qid, "arrays", skill, difficulty, qtype,
                  f"What is printed?\n```java\nint[] nums = {{{values[0]}, {values[1]}, {values[2]}, {values[3]}}};\nSystem.out.println(nums[{idx % 4}]);\n```",
                  str(values[idx % 4]),
                  "Use the index inside brackets to choose the array element.",
                  tags=[skill, "trace_execution"])
    if qtype == "debugging":
        return _q(qid, "arrays", skill, difficulty, qtype,
                  "Find the bug: `for (int i = 0; i <= nums.length; i++) { System.out.println(nums[i]); }`",
                  "use i < nums.length",
                  "The last valid index is `nums.length - 1`, so the loop condition should be `i < nums.length`.",
                  accepted=["i < nums.length", "change <= to <", "off by one"], patterns=["<", "length"], tags=[skill, "off_by_one"])
    return _q(qid, "arrays", skill, difficulty, qtype,
              f"Write one Java statement that creates an int array named `nums` with the values {values[0]}, {values[1]}, and {values[2]}.",
              f"int[] nums = {{{values[0]}, {values[1]}, {values[2]}}};",
              "An int array literal uses braces with comma-separated values.",
              patterns=["int[]", "nums", "{"], tags=[skill])


def _methods(qid: str, difficulty: int, variant: int, qtype: str) -> Dict[str, Any]:
    skill = "method_parameters"
    n = 2 + difficulty + variant % 5
    if qtype == "multiple_choice":
        correct = "return"
        return _q(qid, "methods", skill, difficulty, qtype,
                  "Which keyword sends a value back from a method to its caller?",
                  correct,
                  "The `return` keyword sends a method result back to the place where the method was called.",
                  choices=_shuffle_choices(correct, ["print", "void", "class"], variant), tags=[skill])
    if qtype == "fill_blank":
        return _q(qid, "methods", skill, difficulty, qtype,
                  "Fill in the blank: `public static int addOne(int n) { _____ n + 1; }`",
                  "return",
                  "A non-void method must return a value of the declared type.",
                  tags=[skill])
    if qtype == "code_tracing":
        return _q(qid, "methods", skill, difficulty, qtype,
                  f"What is printed?\n```java\nstatic int doubleIt(int n) {{\n    return n * 2;\n}}\nSystem.out.println(doubleIt({n}));\n```",
                  str(n * 2),
                  "Substitute the argument into the method, compute the return value, then print it.",
                  tags=[skill, "trace_execution"])
    if qtype == "debugging":
        return _q(qid, "methods", skill, difficulty, qtype,
                  "Find the bug: `public static int getFive() { System.out.println(5); }`",
                  "missing return",
                  "An `int` method must return an int value. Printing is not the same as returning.",
                  accepted=["return 5", "add return", "missing return"], patterns=["return"], tags=[skill, "syntax_vs_logic_error"])
    return _q(qid, "methods", skill, difficulty, qtype,
              "Write a Java method header for a method named `square` that takes one int parameter named `n` and returns an int.",
              "public static int square(int n)",
              "The return type is `int`, the method name is `square`, and the parameter is `int n`.",
              accepted=["int square(int n)", "public int square(int n)"], patterns=["int", "square", "int n"], tags=[skill])


def _classes_objects(qid: str, difficulty: int, variant: int, qtype: str) -> Dict[str, Any]:
    skill = "object_state"
    name = ["Dog", "Car", "Book", "Student", "Game"][variant % 5]
    obj = name.lower()
    if qtype == "multiple_choice":
        correct = f"{name} {obj} = new {name}();"
        return _q(qid, "classes_objects", skill, difficulty, qtype,
                  f"Which line creates a new `{name}` object and stores it in variable `{obj}`?",
                  correct,
                  "Creating an object uses `new ClassName()` and stores the reference in a variable.",
                  choices=_shuffle_choices(correct, [f"new {name} {obj};", f"{obj} = {name}();", f"class {obj} = new class();"], variant), tags=[skill])
    if qtype == "fill_blank":
        return _q(qid, "classes_objects", skill, difficulty, qtype,
                  f"Fill in the blank to create an object: `{name} {obj} = _____ {name}();`",
                  "new",
                  "The `new` keyword creates a new object.",
                  tags=[skill])
    if qtype == "code_tracing":
        age = 5 + difficulty + variant
        return _q(qid, "classes_objects", skill, difficulty, qtype,
                  f"What is printed?\n```java\nclass Dog {{ int age = {age}; }}\nDog pet = new Dog();\nSystem.out.println(pet.age);\n```",
                  str(age),
                  "The object `pet` has an `age` field, and the print statement reads that field.",
                  tags=[skill, "trace_execution"])
    if qtype == "debugging":
        return _q(qid, "classes_objects", skill, difficulty, qtype,
                  f"Find the bug: `{name} {obj} = {name}();`",
                  "missing new",
                  "Object creation needs the `new` keyword before the constructor call.",
                  accepted=["add new", "new", "missing new"], patterns=["new"], tags=[skill, "syntax_vs_logic_error"])
    return _q(qid, "classes_objects", skill, difficulty, qtype,
              f"Write one Java statement that creates a `{name}` object named `{obj}` using the no-argument constructor.",
              f"{name} {obj} = new {name}();",
              "Use the class name as the type, then `new ClassName()` to create the object.",
              patterns=[name, obj, "new"], tags=[skill])


def _debugging(qid: str, difficulty: int, variant: int, qtype: str) -> Dict[str, Any]:
    skill = "syntax_vs_logic_error"
    n = 5 + difficulty + variant
    if qtype == "multiple_choice":
        correct = "missing semicolon"
        return _q(qid, "debugging", skill, difficulty, qtype,
                  f"What is wrong with this Java statement? `System.out.println({n})`",
                  correct,
                  "Most Java statements end with a semicolon.",
                  choices=_shuffle_choices(correct, ["wrong keyword", "extra parenthesis", "nothing is wrong"], variant), tags=[skill])
    if qtype == "fill_blank":
        return _q(qid, "debugging", skill, difficulty, qtype,
                  "Fill in the blank: A mistake that stops Java from compiling is usually a _____ error.",
                  "syntax",
                  "Syntax errors break the rules of the Java language, so the compiler rejects them.",
                  accepted=["compile", "compiler"], tags=[skill])
    if qtype == "code_tracing":
        return _q(qid, "debugging", skill, difficulty, qtype,
                  f"What is printed after the bug is fixed?\n```java\nint x = {n};\nSystem.out.println(x);\n```",
                  str(n),
                  "The corrected print statement outputs the current value of x.",
                  tags=[skill, "trace_execution"])
    if qtype == "debugging":
        return _q(qid, "debugging", skill, difficulty, qtype,
                  f"Fix the bug: `int x = {n}`",
                  "add semicolon",
                  "The line should end with a semicolon: `int x = value;`.",
                  accepted=[";", "semicolon", f"int x = {n};"], patterns=[";"], tags=[skill])
    return _q(qid, "debugging", skill, difficulty, qtype,
              f"Rewrite this line correctly: `System.out.println({n})`",
              f"System.out.println({n});",
              "Add the semicolon at the end of the statement.",
              patterns=["System.out.println", str(n), ";"], tags=[skill])


def _code_tracing(qid: str, difficulty: int, variant: int, qtype: str) -> Dict[str, Any]:
    skill = "trace_execution"
    a = 1 + difficulty + variant
    b = 2 + (variant % 5)
    if qtype == "multiple_choice":
        correct = str(a + b)
        return _q(qid, "code_tracing", skill, difficulty, qtype,
                  f"What does this print? `int a = {a}; int b = {b}; System.out.println(a + b);`",
                  correct,
                  "Track the value of each variable, then evaluate the expression inside println.",
                  choices=_shuffle_choices(correct, [str(a), str(b), str(a * b)], variant), tags=[skill])
    if qtype == "fill_blank":
        return _q(qid, "code_tracing", skill, difficulty, qtype,
                  f"Fill in the blank with the printed value: `int x = {a}; x = x + {b}; System.out.println(x);` prints _____ .",
                  str(a + b),
                  "Update x first, then print the new value.",
                  tags=[skill, "variable_assignment"])
    if qtype == "code_tracing":
        result = (a + b) * difficulty
        return _q(qid, "code_tracing", skill, difficulty, qtype,
                  f"What is printed?\n```java\nint a = {a};\nint b = {b};\nint c = (a + b) * {difficulty};\nSystem.out.println(c);\n```",
                  str(result),
                  "Calculate inside the parentheses first, multiply, then print c.",
                  tags=[skill, "operator_precedence"])
    if qtype == "debugging":
        return _q(qid, "code_tracing", skill, difficulty, qtype,
                  f"A student says this prints {a}, but it actually prints something else. What should they track? `int x = {a}; x = x + {b}; System.out.println(x);`",
                  "track the update to x",
                  "The second line changes x before it is printed.",
                  accepted=["x changes", "update x", "track x"], patterns=["x"], tags=[skill, "variable_assignment"])
    return _q(qid, "code_tracing", skill, difficulty, qtype,
              f"What value should you write in a trace table after this line runs? `int total = {a} + {b};`",
              str(a + b),
              "A trace table records the value stored after a line executes.",
              tags=[skill])


_GENERATORS = {
    "variables": _variables,
    "primitive_types": _primitive_types,
    "strings": _strings,
    "operators": _operators,
    "conditionals": _conditionals,
    "loops": _loops,
    "arrays": _arrays,
    "methods": _methods,
    "classes_objects": _classes_objects,
    "debugging": _debugging,
    "code_tracing": _code_tracing,
}


def generate_beginner_question_bank(min_questions: int = 1100) -> List[Dict[str, Any]]:
    questions: list[Dict[str, Any]] = []
    counter = 1
    questions_per_difficulty = max(20, min_questions // (len(BEGINNER_TOPICS) * 5))

    for topic, skill, _desc in BEGINNER_TOPICS:
        generator = _GENERATORS[topic]
        for difficulty in range(1, 6):
            for variant in range(questions_per_difficulty):
                qtype = QUESTION_TYPES[variant % len(QUESTION_TYPES)]
                qid = f"JAVA_{counter:04d}"
                questions.append(generator(qid, difficulty, variant, qtype))
                counter += 1

    # Make sure prompt-level duplicates cannot slip through.
    seen: set[str] = set()
    unique: list[Dict[str, Any]] = []
    for q in questions:
        prompt = str(q.get("prompt", "")).strip()
        if prompt in seen:
            q = dict(q)
            q["prompt"] = f"{prompt}\nScenario {q['question_id']}"
        seen.add(str(q.get("prompt", "")).strip())
        unique.append(q)
    return unique


def generate_beginner_hint_map() -> Dict[str, Any]:
    entries = {
        "variable_assignment": ("Variables store values that can be read or changed later.", ["variables"]),
        "primitive_type_selection": ("Choose a type that matches the kind of data: int, double, boolean, or char.", ["primitive_types"]),
        "string_basics": ("Strings store text and often use methods such as equals and length.", ["strings"]),
        "string_comparison": ("String contents should usually be compared with .equals(...), not ==.", ["strings"]),
        "operator_precedence": ("Java evaluates multiplication before addition unless parentheses change the order.", ["operators"]),
        "operator_basics": ("Operators calculate, compare, or combine boolean values.", ["operators"]),
        "conditional_logic": ("An if statement runs only the branch whose condition is true.", ["conditionals"]),
        "loop_boundaries": ("When a loop starts at 0 and uses < n, it runs n times.", ["loops"]),
        "off_by_one": ("Check whether the final value should be included or excluded.", ["loops", "arrays"]),
        "array_indexing": ("Array indexes start at 0, and the largest valid index is length - 1.", ["arrays"]),
        "method_parameters": ("Arguments go into parameters; return sends a result back.", ["methods"]),
        "object_state": ("Objects are created with new, and fields store object data.", ["classes_objects"]),
        "syntax_vs_logic_error": ("Syntax errors break Java rules; logic errors run but produce the wrong result.", ["debugging"]),
        "trace_execution": ("Trace code in order and update your variable table after each line.", ["code_tracing"]),
    }
    return {
        key: {
            "description": desc,
            "related_topics": topics,
            "diagnostic_signals": [
                f"incorrect answer on a {topics[0].replace('_', ' ')} question",
                "hint used before answering",
                "wrong trace or small syntax repair",
            ],
            "hint_sequence": [
                {"level": 1, "type": "conceptual", "text": desc},
                {"level": 2, "type": "strategic", "text": "Try a tiny example and write values down step by step."},
                {"level": 3, "type": "worked_example", "text": "Read the code from top to bottom, update one value at a time, then answer."},
            ],
            "remediation_activity": f"Give easier {topics[0].replace('_', ' ')} practice before increasing difficulty.",
        }
        for key, (desc, topics) in entries.items()
    }


def generate_beginner_policy() -> Dict[str, Any]:
    return {
        "name": "Local Beginner Java ITS Policy",
        "cold_start": {
            "assumption": "A newly registered learner starts at difficulty 1.",
            "first_session_difficulty": 1,
            "session_length": 20,
        },
        "target_success_probability": {
            "lower": 0.65,
            "upper": 0.80,
            "rationale": "The trained model still ranks same-difficulty candidates so practice stays achievable but not too easy.",
        },
        "session_rules": {
            "questions_per_session": 20,
            "fixed_difficulty_per_session": True,
            "level_up": "17 or more correct out of 20",
            "stay": "12 to 16 correct out of 20",
            "level_down": "fewer than 12 correct out of 20",
            "anti_repetition": "Previously answered question IDs are excluded for the logged-in user.",
        },
        "topic_choice": "Learner can practice a selected topic or mixed random practice.",
    }


def generate_all_beginner_content(output_dir: str | Path, min_questions: int = 1100) -> Dict[str, Path]:
    output_dir = ensure_dir(output_dir)
    qb = generate_beginner_question_bank(min_questions=min_questions)
    paths = {
        "question_bank": write_json(qb, output_dir / "question_bank.json"),
        "misconception_hint_map": write_json(generate_beginner_hint_map(), output_dir / "misconception_hint_map.json"),
        "pedagogical_policy": write_json(generate_beginner_policy(), output_dir / "pedagogical_policy.json"),
    }
    return paths
