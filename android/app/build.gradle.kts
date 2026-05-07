plugins {
    id("com.android.application")
}

android {
    namespace = "com.shiyu.hccr"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.shiyu.hccr"
        minSdk = 24
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0"

        ndk {
            abiFilters += listOf("arm64-v8a", "x86_64")
        }
    }

    buildTypes {
        debug {
            externalNativeBuild {
                cmake {
                    arguments("-DCMAKE_BUILD_TYPE=Release")
                }
            }
        }
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/jni/CMakeLists.txt")
            version = "3.22.1+"
        }
    }

    ndkVersion = "30.0.14904198"
}

dependencies {
    implementation("androidx.core:core:1.15.0")
}
