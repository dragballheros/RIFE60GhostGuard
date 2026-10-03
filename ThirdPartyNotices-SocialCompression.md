# Social Compression Guard model notice

The bundled Social Compression Guard uses **1xDeJPG_realplksr_otf**, a 1× RealPLKSR restoration model by Philip Hofmann.

Purpose: compression-artifact and blur removal without changing image size.

License: **CC BY 4.0**. Attribution is required.

Core ML conversion provenance:
- iOS Core ML export: 333i/1xDeJPG_realplksr_otf-coreml-ios
- Original model: Phips/1xDeJPG_realplksr_otf
- The Core ML package is distributed as a precompiled 512×512 iOS model and is used unchanged.

The app applies the model before RIFE at source cadence, using overlapping 512×512 tiles with feathered blending. This is a compression-restoration stage inspired by the documented behavior of commercial restoration tools; it does not claim to reproduce Topaz Video AI's proprietary model or implementation.
