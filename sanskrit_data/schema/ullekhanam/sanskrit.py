import logging
logging.basicConfig(
  level=logging.DEBUG,
  format="%(levelname)s: %(asctime)s {%(filename)s:%(lineno)d}: %(message)s "
)


from sanskrit_data.schema import common
from sanskrit_data.schema.books import BookPortion
from sanskrit_data.schema.common import Text, Target
from sanskrit_data.schema.ullekhanam import Annotation, TextTarget, TextAnnotation


class PadaAnnotation(Annotation):
  schema = common.recursively_merge_json_schemas(Annotation.schema, ({
    "type": "object",
    "description": "A grammatical pada - subanta or tiNanta.",
    "properties": {
      common.TYPE_FIELD: {
        "enum": ["PadaAnnotation"]
      },
      "targets": {
        "type": "array",
        "items": TextTarget.schema
      },
      "word": Text.schema,
      "root": Text.schema,
    },
  }))

  target_class = TextTarget

  @classmethod
  def get_allowed_target_classes(cls):
    return [BookPortion, TextAnnotation]

  # noinspection PyMethodOverriding
  def set_base_details(self, targets, source, word, root):
    super(PadaAnnotation, self).set_base_details(targets, source)
    # noinspection PyAttributeOutsideInit
    self.word = word
    # noinspection PyAttributeOutsideInit
    self.root = root

  @classmethod
  def from_details(cls, targets, source, word, root):
    annotation = PadaAnnotation()
    annotation.set_base_details(targets, source, word, root)
    annotation.validate()
    return annotation


class SubantaAnnotation(PadaAnnotation):
  schema = common.recursively_merge_json_schemas(PadaAnnotation.schema, ({
    "type": "object",
    "description": "Anything ending with a sup affix. Includes avyaya-s.",
    "properties": {
      common.TYPE_FIELD: {
        "enum": ["SubantaAnnotation"]
      },
      "linga": {
        "type": "string",
        "enum": ["strii", "pum", "napum", "avyaya"]
      },
      "vibhakti": {
        "type": "string",
        "enum": ["1", "2", "3", "4", "5", "6", "7", "1.sambodhana"]
      },
      "vachana": {
        "type": "integer",
        "enum": [1, 2, 3]
      }
    },
  }))

  # noinspection PyMethodOverriding
  @classmethod
  def from_details(cls, targets, source, word, root, linga, vibhakti, vachana):
    obj = SubantaAnnotation()
    obj.set_base_details(targets, source, word, root)
    obj.linga = linga
    obj.vibhakti = vibhakti
    obj.vachana = vachana
    obj.validate()
    return obj


class TinantaAnnotation(PadaAnnotation):
  schema = common.recursively_merge_json_schemas(PadaAnnotation.schema, ({
    "type": "object",
    "description": "Anything ending with a tiN affix.",
    "properties": {
      common.TYPE_FIELD: {
        "enum": ["TinantaAnnotation"]
      },
      "lakAra": {
        "type": "string",
        "enum": ["laT", "laN", "vidhi-liN", "AshIr-liN", "loT", "liT", "luT", "LT", "luN", "LN", "leT"]
      },
      "puruSha": {
        "type": "string",
        "enum": ["prathama", "madhyama", "uttama"]
      },
      "vachana": {
        "type": "integer",
        "enum": [1, 2, 3]
      }
    },
  }))

  # noinspection PyMethodOverriding
  @classmethod
  def from_details(cls, targets, source, word, root, lakAra, puruSha, vachana):
    obj = TinantaAnnotation()
    obj.set_base_details(targets, source, word, root)
    obj.lakAra = lakAra
    obj.puruSha = puruSha
    obj.vachana = vachana
    obj.validate()
    return obj


class TextSambandhaAnnotation(Annotation):
  schema = common.recursively_merge_json_schemas(Annotation.schema, ({
    "type": "object",
    "description": "Describes connection between two text portions. Such connection is directional (ie it connects words in a source sentence to words in a target sentence.)",
    "properties": {
      common.TYPE_FIELD: {
        "enum": ["TextSambandhaAnnotation"]
      },
      "targets": {
        "description": "A pair of texts being connected. First text is the 'source text', second is the 'target text'",
      },
      "category": {
        "type": "string"
      },
      "source_text_padas": {
        "type": "array",
        "description": "The entity being annotated.",
        "items": Target.schema,
        "minItems": 1,
      },
      "target_text_padas": {
        "type": "array",
        "description": "The entity being annotated.",
        "minItems": 1,
        "items": Target.schema
      }
    },
    "required": ["combined_string"]
  }))

  def validate(self, db_interface=None, user=None):
    super(TextSambandhaAnnotation, self).validate(db_interface=db_interface, user=user)
    Target.check_target_classes(targets_to_check=self.source_text_padas, allowed_types=[PadaAnnotation], db_interface=db_interface, targeting_obj=self)
    Target.check_target_classes(targets_to_check=self.target_text_padas, allowed_types=[PadaAnnotation], db_interface=db_interface, targeting_obj=self)

  @classmethod
  def get_allowed_target_classes(cls):
    return [BookPortion, TextAnnotation]


class SandhiAnnotation(Annotation):
  schema = common.recursively_merge_json_schemas(Annotation.schema, ({
    "type": "object",
    "properties": {
      common.TYPE_FIELD: {
        "enum": ["SandhiAnnotation"]
      },
      "combined_string": Text.schema,
      "sandhi_type": {
        "type": "string"
      }
    },
    "required": ["combined_string"]
  }))

  @classmethod
  def get_allowed_target_classes(cls):
    return [PadaAnnotation]

  @classmethod
  def from_details(cls, targets, source, combined_string, sandhi_type="UNK"):
    annotation = SandhiAnnotation()
    annotation.set_base_details(targets, source)
    annotation.combined_string = combined_string
    annotation.sandhi_type = sandhi_type
    annotation.validate()
    return annotation


class SamaasaAnnotation(Annotation):
  schema = common.recursively_merge_json_schemas(Annotation.schema, ({
    "type": "object",
    "properties": {
      common.TYPE_FIELD: {
        "enum": ["SamaasaAnnotation"]
      },
      "component_padas": {
        "type": "array",
        "description": "Pointers to PadaAnnotation objects corresponding to components of the samasta-pada",
        "minItems": 1,
        "items": Target.schema
      },
      "samaasa_type": {
        "type": "string"
      }
    },
  }))

  @classmethod
  def get_allowed_target_classes(cls):
    return [PadaAnnotation]

  def validate(self, db_interface=None, user=None):
    super(SamaasaAnnotation, self).validate(db_interface=db_interface, user=user)
    Target.check_target_classes(targets_to_check=self.component_padas, allowed_types=[PadaAnnotation], db_interface=db_interface, targeting_obj=self)

  @classmethod
  def from_details(cls, targets, source, combined_string, samaasa_type="UNK"):
    annotation = SamaasaAnnotation()
    annotation.set_base_details(targets, source)
    annotation.combined_string = combined_string
    annotation.type = samaasa_type
    annotation.validate()
    return annotation


import sys
# Essential for depickling to work.
common.update_json_class_index(sys.modules[__name__])
logging.debug(common.json_class_index)
