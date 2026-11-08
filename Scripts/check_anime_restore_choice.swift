import Foundation

func require(_ condition: Bool, _ message: String) {
    if !condition { fatalError(message) }
}

require(AnimeRestoreChoice.tileCount(length: 1280) == 6, "1280 tile count")
require(AnimeRestoreChoice.tileCount(length: 720) == 3, "720 tile count")
require(AnimeRestoreChoice.tileCount(length: 1280) * AnimeRestoreChoice.tileCount(length: 720) == 18, "1280x720 tiles")
let under = AnimeRestoreChoice.pick(quality: .auto, apisrMillisecondsPerFrame: 1800)
require(under.0 == .apisr && under.1 == 900, "APISR within 1s after 2s reuse")
let over = AnimeRestoreChoice.pick(quality: .auto, apisrMillisecondsPerFrame: 2200)
require(over.0 == .span && over.1 == 1100, "SPAN when APISR is over the budget")
require(AnimeRestoreChoice.pick(quality: .span, apisrMillisecondsPerFrame: 100).0 == .span, "fast picker")
require(AnimeRestoreChoice.pick(quality: .apisr, apisrMillisecondsPerFrame: 9000).0 == .apisr, "max picker")
print("ANIME_RESTORE_CHOICE_PASS")
