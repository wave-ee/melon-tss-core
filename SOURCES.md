# Source provenance

Regression architecture and prediction aggregation were extracted from the
saved `padf_historical_protocol_balanced_20261004` experiment source snapshot.
The segmentation modules and physical descriptors were extracted from the
corresponding project implementations. The source list below records the
provenance of the modules represented in this repository.

```json
[
  {
    "component": "unet",
    "source": "unet.py",
    "sha256": "d8a75381e9d167e4ae92447789ac1eb073379ff6a5936cb18a04293b43062874"
  },
  {
    "component": "cldice",
    "source": "cldice_loss.py",
    "sha256": "1829984a9ad93806024b80503cc3bc2d48b059461d4dc1d1c0a981ffb87030c3"
  },
  {
    "component": "seg_config",
    "source": "da_unet_toposeg_netcldice_finetune_4k_melon.py",
    "sha256": "23a8b00d7e2204c09c75f01debb12f5f4d41ccce723be1bcc654690cdc732ce8"
  },
  {
    "component": "phenotypes",
    "source": "extract_v17_physics.py",
    "sha256": "e5f51474570fb28a29f20d99eafd3152d467af50a7f9b7e41bc72e84cf9359de"
  },
  {
    "component": "fusion",
    "source": "physics_assisted_region_common.py",
    "sha256": "f1f13613d7d3aae952ed19d0ee50235e565f509e283f679ebf242fe64dbf2680"
  },
  {
    "component": "backbone",
    "source": "end2end_region_aware_regression.py",
    "sha256": "e51caefd4247078ee554e1887c097d6ceedc509a0a0b6d52b0959ba528d87866"
  },
  {
    "component": "runner",
    "source": "run_historical_balanced.py",
    "sha256": "9228de8f68520bdfbbf902f8baf490cea47af4a4166d52232797f885df842135"
  },
  {
    "component": "contract",
    "source": "run_contract.json",
    "sha256": "cd63af43d2419d0628300a9e9062b5d940788fafd6400f79bbddc1aa5cb89910"
  }
]
```

Dependencies and concepts:

- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation), Apache-2.0;
  the fixed integration targets version 1.2.2.
- [torchvision](https://github.com/pytorch/vision), ConvNeXt-Tiny features.
- Shit et al., *clDice: A Novel Topology-Preserving Loss Function for Tubular
  Structure Segmentation*, CVPR 2021.
- [MelonDataset](https://github.com/codaibk/MelonDataset), separate dataset source.
