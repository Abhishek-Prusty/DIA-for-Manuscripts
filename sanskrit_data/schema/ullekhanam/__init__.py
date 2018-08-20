# -*- coding: utf-8 -*-
"""
Intro
-----------

-  Annotations are stored in a directed acyclic graph, for example - a book portion having a TextAnnotation having PadaAnnotations having SamaasaAnnotations.

    -  Some Annotations (eg. SandhiAnnotation, TextAnnotation) can
       have multiple "targets" (ie. other objects being annotated).
    -  Rather than a simple tree, we end up with a Directed Acyclic
       Graph (DAG) of Annotation objects.

-  JSON schema mindmap
   `here <https://drive.mindmup.com/map?state=%7B%22ids%22:%5B%220B1_QBT-hoqqVbHc4QTV3Q2hjdTQ%22%5D,%22action%22:%22open%22,%22userId%22:%22109000762913288837175%22%7D>`__
   (Updated as needed).
- For general context and class diagram, refer to :mod:`~sanskrit_data.schema`.

"""
import logging
import sys

from sanskrit_data.schema import common
from sanskrit_data.schema.books import BookPortion, CreationDetails
from sanskrit_data.schema.common import JsonObject, UllekhanamJsonObject, Target, DataSource, Text, NamedEntity

logging.basicConfig(
  level=logging.DEBUG,
  format="%(levelname)s: %(asctime)s {%(filename)s:%(lineno)d}: %(message)s "
)


class Annotation(UllekhanamJsonObject):
  schema = common.recursively_merge_json_schemas(UllekhanamJsonObject.schema, ({
    "type": "object",
    "properties": {
      common.TYPE_FIELD: {
        "enum": ["Annotation"]
      },
      "targets": {
        "minItems": 1,
      },
    },
    "required": ["targets", "source"]
  }))

  def __init__(self):
    super(Annotation, self).__init__()

  @classmethod
  def get_allowed_target_classes(cls):
    return [BookPortion, Annotation]

  def set_base_details(self, targets, source):
    # noinspection PyAttributeOutsideInit
    self.targets = targets
    # noinspection PyAttributeOutsideInit
    self.source = source


class Rectangle(JsonObject):
  schema = common.recursively_merge_json_schemas(JsonObject.schema, ({
    "type": "object",
    "description": "A rectangle within an image.",
    "properties": {
      common.TYPE_FIELD: {
        "enum": ["Rectangle"]
      },
      "x1": {
        "type": "integer"
      },
      "y1": {
        "type": "integer"
      },
      "w": {
        "type": "integer"
      },
      "h": {
        "type": "integer"
      },
    },
    "required": ["x1", "y1", "w", "h"]
  }))

  @classmethod
  def from_details(cls, x=-1, y=-1, w=-1, h=-1, score=0.0):
    rectangle = Rectangle()
    rectangle.x1 = x
    rectangle.y1 = y
    rectangle.w = w
    rectangle.h = h
    rectangle.score = score
    rectangle.validate()
    return rectangle

  # Two (segments are 'equal' if they overlap
  def __eq__(self, other):
    xmax = max(self.x, other.x)
    ymax = max(self.y, other.y)
    overalap_w = min(self.x + self.w, other.x + other.w) - xmax
    overalap_h = min(self.y + self.h, other.y + other.h) - ymax
    return overalap_w > 0 and overalap_h > 0

  def __ne__(self, other):
    return not self.__eq__(other)

  # noinspection PyTypeChecker
  def __cmp__(self, other):
    if self == other:
      logging.info(str(self) + " overlaps " + str(other))
      return 0
    elif (self.y < other.y) or ((self.y == other.y) and (self.x < other.x)):
      return -1
    else:
      return 1


# noinspection PyMethodOverriding
class ImageTarget(Target):
  schema = common.recursively_merge_json_schemas(Target.schema, ({
    "type": "object",
    "description": "The rectangle within the image being targetted.",
    "properties": {
      common.TYPE_FIELD: {
        "enum": ["ImageTarget"]
      },
      "rectangle": Rectangle.schema
    },
    "required": ["rectangle"]
  }))

  # TODO use w, h instead.
  # noinspection PyMethodOverriding
  @classmethod
  def from_details(cls, container_id, rectangle):
    target = ImageTarget()
    target.container_id = container_id
    target.rectangle = rectangle
    target.validate()
    return target


class ValidationAnnotationSource(DataSource):
  """We don't override the schema here as no new fields are added."""
  def setup_source(self, db_interface=None, user=None):
    self.infer_by_admin(db_interface=db_interface, user=user)
    super(ValidationAnnotationSource, self).setup_source(db_interface=db_interface, user=user)


class ValidationAnnotation(Annotation):
  schema = common.recursively_merge_json_schemas(Annotation.schema, ({
    "type": "object",
    "description": "Any user can validate a certain annotation (or other object). But it is up to various systems whether such 'validation' has any effect.",
    "properties": {
      common.TYPE_FIELD: {
        "enum": ["ValidationAnnotation"]
      },
      "source": ValidationAnnotationSource.schema
    },
  }))

  def __init__(self):
    super(ValidationAnnotation, self).__init__()
    self.source = ValidationAnnotationSource()


class ImageAnnotation(Annotation):
  """ Mark a certain fragment of an image.

  `An introductory video <https://www.youtube.com/watch?v=SHzD3f5nPt0&t=29s>`_
  """
  schema = common.recursively_merge_json_schemas(Annotation.schema, ({
    "type": "object",
    "description": "A rectangle within an image, picked by a particular annotation source.",
    "properties": {
      common.TYPE_FIELD: {
        "enum": ["ImageAnnotation"]
      },
      "targets": {
        "type": "array",
        "items": ImageTarget.schema
      }
    },
  }))

  target_class = ImageTarget

  @classmethod
  def get_allowed_target_classes(cls):
    return [BookPortion, ImageAnnotation]

  @classmethod
  def from_details(cls, targets, source):
    annotation = ImageAnnotation()
    annotation.set_base_details(targets, source)
    annotation.validate()
    return annotation


# Targets: ImageAnnotation(s) or  TextAnnotation or BookPortion
class TextAnnotation(Annotation):
  schema = common.recursively_merge_json_schemas(Annotation.schema, ({
    "type": "object",
    "description": "Annotation of some (sub)text from within the object (image or another text) being annotated. Tells: 'what is written in this image? or text portion?",
    "properties": {
      common.TYPE_FIELD: {
        "enum": ["TextAnnotation"]
      },
      "content": Text.schema,
    },
    "required": ["content"]
  }))

  @classmethod
  def get_allowed_target_classes(cls):
    return [BookPortion, ImageAnnotation]

  @classmethod
  def from_details(cls, targets, source, content):
    annotation = TextAnnotation()
    annotation.set_base_details(targets, source)
    annotation.content = content
    annotation.validate()
    return annotation

  @classmethod
  def add_indexes(cls, db_interface):
    super(TextAnnotation, cls).add_indexes(db_interface=db_interface)
    db_interface.add_index(keys_dict={
      "content.search_strings": 1
    }, index_name="content_search_strings")


class CommentAnnotation(TextAnnotation):
  schema = common.recursively_merge_json_schemas(TextAnnotation.schema, ({
    "description": "A comment that can be associated with nearly any Annotation or BookPortion.",
    "properties": {
      common.TYPE_FIELD: {
        "enum": ["CommentAnnotation"]
      },
    }
  }))

  @classmethod
  def get_allowed_target_classes(cls):
    return [BookPortion, Annotation]


class TranslationAnnotation(TextAnnotation):
  schema = common.recursively_merge_json_schemas(TextAnnotation.schema, ({
    "description": "A comment that can be associated with nearly any Annotation or BookPortion.",
    "properties": {
      common.TYPE_FIELD: {
        "enum": ["TranslationAnnotation"]
      },
    }
  }))

  @classmethod
  def get_allowed_target_classes(cls):
    return [BookPortion, Annotation]


class QuoteAnnotation(TextAnnotation):
  schema = common.recursively_merge_json_schemas(TextAnnotation.schema, ({
    "description": "A quote, a memorable text fragment.",
    "properties": {
      common.TYPE_FIELD: {
        "enum": ["QuoteAnnotation"]
      },
      "editable_by_others": {
        "default": False
      },
    }
  }))

  @classmethod
  def get_allowed_target_classes(cls):
    return [BookPortion, Annotation]


class Metre(NamedEntity):
  schema = common.recursively_merge_json_schemas(NamedEntity.schema, ({
    "type": "object",
    "properties": {
      common.TYPE_FIELD: {
        "enum": ["Metre"]
      }
    }
  }))


class MetreAnnotation(Annotation):
  schema = common.recursively_merge_json_schemas(Annotation.schema, ({
    "description": "A metre, which may be ",
    "properties": {
      common.TYPE_FIELD: {
        "enum": ["MetreAnnotation"]
      },
      "metre": Metre.schema
    }
  }))


class TextOffsetAddress(JsonObject):
  schema = common.recursively_merge_json_schemas(JsonObject.schema, {
    "type": "object",
    "description": "A way to specify a substring.",
    "properties": {
      common.TYPE_FIELD: {
        "enum": ["TextOffsetAddress"]
      },
      "start": {
        "type": "integer"
      },
      "end": {
        "type": "integer"
      }
    }})

  @classmethod
  def from_details(cls, start, end):
    obj = TextOffsetAddress()
    obj.start = start
    obj.end = end
    obj.validate()
    return obj


class TextTarget(Target):
  schema = common.recursively_merge_json_schemas(Target.schema, ({
    "type": "object",
    "description": "A way to specify a particular substring within a string.",
    "properties": {
      common.TYPE_FIELD: {
        "enum": ["TextTarget"]
      },
      "shabda_id": {
        "type": "string",
        "description": "Format: pada_index.shabda_index or just pada_index."
                       "Suppose that some shabda in 'rāgādirogān satatānuṣaktān' is being targetted. "
                       "This has the following pada-vigraha: rāga [comp.]-ādi [comp.]-roga [ac.p.m.]  satata [comp.]-anuṣañj [ac.p.m.]."
                       "Then, rāga has the id 1.1. roga has id 1.3. satata has the id 2.1."
      },
      "offset_address": TextOffsetAddress.schema
    },
  }))

  @classmethod
  def from_details(cls, container_id, shabda_id=None, offset_address=None):
    target = TextTarget()
    target.container_id = container_id
    if shabda_id is not None:
      target.shabda_id = shabda_id
    if offset_address is not None:
      target.offset_address = offset_address
    target.validate()
    return target


# noinspection PyMethodOverriding
# Targets: TextTarget pointing to TextAnnotation
# noinspection PyMethodOverriding
# noinspection PyMethodOverriding,PyPep8Naming
# Targets: a pair of textAnnotation or BookPortion objects
# Targets: two or more PadaAnnotations
# Targets: one PadaAnnotation (the samasta-pada)


class OriginAnnotation(Annotation):
  """See schema.description."""
  schema = common.recursively_merge_json_schemas(Annotation.schema, ({
    "type": "object",
    "description": "A given text may be quoted from some other book. This annotation helps specify such origin.",
    "properties": {
      common.TYPE_FIELD: {
        "enum": ["OriginAnnotation"]
      },
      "originDetails": CreationDetails.schema,
    },
  }))


class Topic(NamedEntity):
  schema = common.recursively_merge_json_schemas(NamedEntity.schema, ({
    "type": "object",
    "properties": {
      common.TYPE_FIELD: {
        "enum": ["Topic"]
      }
    }
  }))


class TopicAnnotation(Annotation):
  """See schema.description."""
  schema = common.recursively_merge_json_schemas(Annotation.schema, ({
    "type": "object",
    "description": "A given text may be quoted from some other book. This annotation helps specify such origin.",
    "properties": {
      common.TYPE_FIELD: {
        "enum": ["TopicAnnotation"]
      },
      "topic": Topic.schema,
    },
  }))


class RatingAnnotation(Annotation):
  """See schema.description."""
  schema = common.recursively_merge_json_schemas(Annotation.schema, ({
    "type": "object",
    "description": "A given text may be quoted from some other book. This annotation helps specify such origin.",
    "properties": {
      common.TYPE_FIELD: {
        "enum": ["RatingAnnotation"]
      },
      "rating": {
        "type": "number"
      },
      "editable_by_others": {
        "type": "boolean",
        "description": "Can this annotation be taken over by others for wiki-style editing or deleting?",
        "default": False
      }
    },
  }))


# Essential for depickling to work.
common.update_json_class_index(sys.modules[__name__])
logging.debug(common.json_class_index)
