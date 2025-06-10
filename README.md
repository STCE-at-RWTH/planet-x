# Planet X
Some facts:
- I'm bad a project management, and worse at source control. 
- Notebook files can generate really large diffs. 

A conclusion:

Notebook files should not be in the main project repository.

## Shock AD Notebooks

These notebooks usually need a data file, which they will search the `data` directory for. Let me know if you need certain setups written. The scripts to generate these files live in the `Euler2D` project repository.

## Pluto Notebooks

These need Pluto.jl and access to the STCE package registry. Both can be added at the Julia REPL:

```
]registry add https://github.com/STCE-at-RWTH/STCEJuliaRegistry
```

and

```
]add Pluto
```

Pluto can be started from the terminal with `julia -e "using Pluto; Pluto.run(enable_ai_editor_features=false)"`. I guess you could turn on the LLM features, if you really felt like it.

## `data` and `gfx`

Some notebooks need data files, some generate graphics for later.
