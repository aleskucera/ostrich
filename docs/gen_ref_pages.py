"""Generate the code reference pages and navigation."""

from pathlib import Path
import ast
import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

src = Path(__file__).parent.parent / "src"
root = src.parent
members = []

for path in sorted(src.rglob("*.py")):
    module_path = path.relative_to(src).with_suffix("")
    doc_path = path.relative_to(src).with_suffix(".md")
    full_doc_path = "reference" / doc_path

    parts = tuple(module_path.parts)

    if parts[-1] == "__init__":
        continue
    elif parts[-1] == "__main__":
        continue
    else:
        with open(src / ("/".join(parts) + ".py"), "r") as f:
            tree = ast.parse(f.read())
            if ast.get_docstring(tree) is None:
                continue

    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        ident = ".".join(parts)
        members.append(ident)
        fd.write(f"::: {ident}\n\toptions:\n\t\tshow_signature: false")

    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))

doc_path = "ostrich/index.md"
full_doc_path = "reference" + "/" + doc_path
nav["ostrich"] = doc_path
with mkdocs_gen_files.open(full_doc_path, "w") as fd:
    for member in members:
        fd.write(f"::: {member}\n\toptions:\n\t\tshow_signature: false\n\n")
mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())