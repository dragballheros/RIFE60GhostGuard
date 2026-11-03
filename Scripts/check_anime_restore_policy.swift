import Foundation

func require(_ condition: Bool, _ message: String) {
    if !condition { fatalError(message) }
}

require(AnimeRestorePolicy.shortSideQualifies(width: 1280, height: 720), "720p landscape")
require(AnimeRestorePolicy.shortSideQualifies(width: 720, height: 1280), "720p portrait")
require(AnimeRestorePolicy.shortSideQualifies(width: 960, height: 540), "540p")
require(AnimeRestorePolicy.shortSideQualifies(width: 854, height: 480), "480p")
require(!AnimeRestorePolicy.shortSideQualifies(width: 1920, height: 1080), "1080p stays off")
require(!AnimeRestorePolicy.shortSideQualifies(width: 1080, height: 1920), "1080p portrait stays off")
require(!AnimeRestorePolicy.shortSideQualifies(width: 0, height: 720), "unknown size")

let auto = AnimeRestorePolicy.decide(width: 1280, height: 720, persistedEnabled: false, suppressAutoForThisSource: false, modelRegistered: true)
require(auto.enabled && auto.automatic, "auto enables when a model exists")
let autoNoModel = AnimeRestorePolicy.decide(width: 1280, height: 720, persistedEnabled: false, suppressAutoForThisSource: false, modelRegistered: false)
require(!autoNoModel.enabled && autoNoModel.automatic, "auto without a model leaves pixels unchanged")
let suppressed = AnimeRestorePolicy.decide(width: 1280, height: 720, persistedEnabled: false, suppressAutoForThisSource: true, modelRegistered: true)
require(!suppressed.enabled && !suppressed.automatic, "explicit off blocks auto for this source")
let userOn = AnimeRestorePolicy.decide(width: 1920, height: 1080, persistedEnabled: true, suppressAutoForThisSource: false, modelRegistered: true)
require(userOn.enabled && !userOn.automatic, "persisted on applies above 720")
let userOnNoModel = AnimeRestorePolicy.decide(width: 1920, height: 1080, persistedEnabled: true, suppressAutoForThisSource: false, modelRegistered: false)
require(!userOnNoModel.enabled, "persisted on without a model does not rewrite frames")
let off = AnimeRestorePolicy.decide(width: 1920, height: 1080, persistedEnabled: false, suppressAutoForThisSource: false, modelRegistered: true)
require(!off.enabled, "1080p with toggle off is unchanged")
require(AnimeRestoreRegistry.factory == nil, "no model is registered")
require(!AnimeRestoreRegistry.isRegistered, "registry stays empty")

print("ANIME_RESTORE_POLICY_PASS")
