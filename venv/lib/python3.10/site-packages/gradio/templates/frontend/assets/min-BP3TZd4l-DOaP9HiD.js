import { m as baseIteratee, e as arrayMap, y as baseEach, c as baseFlatten, v as isSymbol } from './_baseUniq-CrYdfo_J-C1cWc0s3.js';
import { aE as isArray, aD as isArrayLike, aF as identity } from './mermaid.core-vMMZVCDT-CT2eqvo9.js';

function flatten(array) {
  var length = array == null ? 0 : array.length;
  return length ? baseFlatten(array) : [];
}
function baseMap(collection, iteratee) {
  var index = -1, result = isArrayLike(collection) ? Array(collection.length) : [];
  baseEach(collection, function(value, key, collection2) {
    result[++index] = iteratee(value, key, collection2);
  });
  return result;
}
function map(collection, iteratee) {
  var func = isArray(collection) ? arrayMap : baseMap;
  return func(collection, baseIteratee(iteratee));
}
function baseLt(value, other) {
  return value < other;
}
function baseExtremum(array, iteratee, comparator) {
  var index = -1, length = array.length;
  while (++index < length) {
    var value = array[index], current = iteratee(value);
    if (current != null && (computed === void 0 ? current === current && !isSymbol(current) : comparator(current, computed))) {
      var computed = current, result = value;
    }
  }
  return result;
}
function min(array) {
  return array && array.length ? baseExtremum(array, identity, baseLt) : void 0;
}

export { min as a, baseMap as b, baseExtremum as c, baseLt as d, flatten as f, map as m };
//# sourceMappingURL=min-BP3TZd4l-DOaP9HiD.js.map
