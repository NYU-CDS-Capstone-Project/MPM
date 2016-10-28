# weinberg-exp

## Instructions:

this workflow generates a e+ e- > µ+ µ- events with polarized beams using madgraph and converts them to flat jsonlines files. If you cannot install `yadage` you can use the Docker image as shown below (This needs Docker installed, see documentation here: https://docs.docker.com/)

    export TOP=https://raw.githubusercontent.com/lukasheinrich/weinberg-exp/master/example_yadage
    eval "$(curl https://raw.githubusercontent.com/diana-hep/yadage/master/yadagedocker.sh)"
    yadage-run -t $TOP workdir rootflow.yml -p nevents=25000 -p seeds=[1,2,3,4] -a $TOP/input.zip \
               -p runcardtempl=run_card.templ -p proccardtempl=sm_proc_card.templ \
               -p sqrtshalf=45 -p polbeam1=0 -p polbeam2=0 

With yadage installed (`pip install yadage`) you can skip the `eval` line.

## Options

Polarization: `polbeam1` and `polbeam2` control the beam polarizations and can be varies from -100 to 100 each.

`nevents`: number of events to generate per seed
`seeds`: Array of seeds for random number generation

The number of events generated will be `(nevents) x len(seeds)`, i.e. in the examples above 100k events will be generated along four parallel production chains (each with a different seed)

## Example Output

The resulting file will be in `{workdir}/merge/out.jsonl`. Below we show a single event to illustrate the resulting JSON structure

        {
      "particles": [
        {
          "status": -1,
          "e": 45,
          "mother1": 0,
          "mother2": 0,
          "pz": 45,
          "px": 0,
          "py": 0,
          "m": 0,
          "color1": 0,
          "color2": 0,
          "lifetime": 0,
          "spin": 1,
          "id": -11
        },
        {
          "status": -1,
          "e": 45,
          "mother1": 0,
          "mother2": 0,
          "pz": -45,
          "px": -0,
          "py": -0,
          "m": 0,
          "color1": 0,
          "color2": 0,
          "lifetime": 0,
          "spin": -1,
          "id": 11
        },
        {
          "status": 2,
          "e": 90,
          "mother1": 1,
          "mother2": 2,
          "pz": 0,
          "px": 0,
          "py": 0,
          "m": 90,
          "color1": 0,
          "color2": 0,
          "lifetime": 0,
          "spin": 0,
          "id": 23
        },
        {
          "status": 1,
          "e": 45,
          "mother1": 3,
          "mother2": 3,
          "pz": -20.465267665,
          "px": -39.899107714,
          "py": 3.7728004224,
          "m": 0,
          "color1": 0,
          "color2": 0,
          "lifetime": 0,
          "spin": -1,
          "id": -13
        },
        {
          "status": 1,
          "e": 45,
          "mother1": 3,
          "mother2": 3,
          "pz": 20.465267665,
          "px": 39.899107714,
          "py": -3.7728004224,
          "m": 0,
          "color1": 0,
          "color2": 0,
          "lifetime": 0,
          "spin": 1,
          "id": 13
        }
      ],
      "eventinfo": {
        "scale": 90,
        "weight": 0.098172,
        "pid": 1,
        "nparticles": 5,
        "aqed": 0.007546771,
        "aqcd": 0.1182338
      }
    }

